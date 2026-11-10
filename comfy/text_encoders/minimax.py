"""MiniMax H3 text/vision conditioning: Qwen3-VL-32B (truncated to 50 layers).

The H3 presentation is NOT chat-templated: token ids are raw prompt/label text
(no special tokens) with explicit vision blocks spliced in:

  t2va:   <prompt>
  fl2va:  "<Picture 1>: " <vision block> ["<Picture 2>: " <vision block>] <prompt>
  ref2va: per condition in request order (1-based ordinals per type):
            image -> "<Picture i>: " <vision block>
            audio -> "<Audio j>: "              (audio never enters Qwen)
            video -> "<Video k>: " then per 2-frame temporal block
                     "<T.T seconds>" <vision block(2 frames)>
          then <prompt>

The conditioning is the unnormalized hidden state after LM layer 50 (the
converted checkpoint is truncated there, so this is simply the last-layer
output with no final norm). Vision-pad positions carry adaLN token tag 0
(video modality) in the DiT; text positions carry tag 1 — the tags are
returned alongside the embeddings as "minimax_token_tags".
"""

import dataclasses
import functools
import logging
import math

import torch
import torch.nn.functional as F
import comfy.ops
import comfy.model_management
import comfy.multigpu
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout
import comfy.sd1_clip
from .qwen3vl import Qwen3VL, Qwen3VLSDTokenizer
from .qwen_vl import smart_resize

VISION_START = 151652
VISION_END = 151653
# FL2VA/Ref2VA tokenizer_config extends Qwen with these, ids fixed by the released tokenizer
MINIMAX_EXTRA_TOKENS = {"<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671,
                        "<|lyrics_start|>": 151672, "<|lyrics_end|>": 151673,
                        "<|caption_start|>": 151674, "<|caption_end|>": 151675}
QWEN_IMAGE_MEAN = [0.5, 0.5, 0.5]
QWEN_IMAGE_STD = [0.5, 0.5, 0.5]
# p-h3-image: Qwen3-VL-4B hidden states read by the text connector, the last one normed
PH3_IMAGE_HIDDEN_STATES = [13, 19, 25, 31, 36]
PH3_IMAGE_TEXT_DIM = 5120


def process_video_block(frames, patch_size=16, temporal_patch_size=2, merge_size=2,
                        min_pixels=3136, max_pixels=12845056):
    """[2, H, W, C] frame pair -> (flatten_patches, grid_thw) with grid_t=1.

    Same resize/normalize policy as process_qwen2vl_images, but the two frames
    fill the temporal patch instead of repeating a single frame.
    """
    t, height, width, _ = frames.shape
    imgs = frames.permute(0, 3, 1, 2)
    h_bar, w_bar = smart_resize(height, width, patch_size * merge_size, min_pixels, max_pixels)

    imgs = F.interpolate(imgs, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(QWEN_IMAGE_MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(QWEN_IMAGE_STD, device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    patches = imgs.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size,
                           patch_size, grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.stack([torch.tensor([1, grid_h, grid_w], device=frames.device, dtype=torch.long)])
    return flatten, grid_thw


def token_tags_from_embeds_info(seq_len, embeds_info):
    # whole vision block VIDEO(0), including the flanking <|vision_start|>/<|vision_end|> tokens
    # embeds_info spans cover only the expanded embeddings, so widen by one on each side.
    tags = torch.ones(seq_len, dtype=torch.long)
    for e in embeds_info:
        if e.get("type") == "image":
            tags[max(0, e["index"] - 1):e["index"] + e["size"] + 1] = 0
    return tags


def _slice_tp_weight(weight, world_size, rank, dim):
    if weight.shape[dim] % world_size != 0:
        raise ValueError(
            f"Cannot tensor-parallel shard weight {tuple(weight.shape)} on dim {dim} "
            f"across {world_size} ranks"
        )
    shard_size = weight.shape[dim] // world_size
    shard_start = rank * shard_size

    if isinstance(weight, QuantizedTensor):
        if weight._layout_cls != "TensorWiseINT8Layout":
            raise TypeError(f"Text TP supports TensorWiseINT8Layout, got {weight._layout_cls}")
        if dim == 1 and getattr(weight._params, "convrot", False):
            group_size = getattr(weight._params, "convrot_groupsize", 256)
            if shard_size % group_size != 0:
                raise ValueError(f"TP input shard {shard_size} is not aligned to ConvRot group size {group_size}")
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        qdata = qdata.narrow(dim, shard_start, shard_size).clone()
        if dim == 0 and scale.numel() == weight.shape[0]:
            scale = scale.reshape(weight.shape[0], -1).narrow(0, shard_start, shard_size).clone()
        else:
            scale = scale.contiguous()
        params = dataclasses.replace(weight._params, scale=scale, orig_shape=tuple(qdata.shape))
        return QuantizedTensor(qdata, weight._layout_cls, params)

    return weight.detach().narrow(dim, shard_start, shard_size).clone()


def _shard_tp_linear(linear, world_size, rank, dim):
    weight = _slice_tp_weight(linear.weight, world_size, rank, dim)
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    if dim == 0:
        linear.out_features = weight.shape[0]
        if linear.bias is not None:
            bias_size = linear.bias.shape[0] // world_size
            linear.bias = torch.nn.Parameter(
                linear.bias.detach().narrow(0, rank * bias_size, bias_size).contiguous(),
                requires_grad=False,
            )
    else:
        linear.in_features = weight.shape[1]
        if linear.bias is not None and rank != 0:
            linear.register_parameter("bias", None)
    if hasattr(linear, "_orig_shape"):
        linear._orig_shape = tuple(weight.shape)


def _resize_tp_linear_topology(linear, world_size, rank, dim):
    out_features = linear.out_features
    in_features = linear.in_features
    if dim == 0:
        if out_features % world_size != 0:
            raise ValueError(f"TP output size {out_features} is not divisible by {world_size}")
        out_features //= world_size
    else:
        if in_features % world_size != 0:
            raise ValueError(f"TP input size {in_features} is not divisible by {world_size}")
        in_features //= world_size

    linear.out_features = out_features
    linear.in_features = in_features
    weight = getattr(linear, "weight", None)
    if weight is not None:
        linear.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device="meta", dtype=weight.dtype),
            requires_grad=False,
        )
    bias = getattr(linear, "bias", None)
    if dim == 0 and bias is not None:
        linear.bias = torch.nn.Parameter(
            torch.empty((out_features,), device="meta", dtype=bias.dtype),
            requires_grad=False,
        )
    elif dim == 1 and bias is not None and rank != 0:
        linear.register_parameter("bias", None)
    if hasattr(linear, "_orig_shape"):
        linear._orig_shape = (out_features, in_features)


def _minimax_h3_language_model(model):
    clip_model = getattr(model, "qwen3vl_32b", None)
    if not isinstance(clip_model, MiniMaxH3ClipModel):
        raise TypeError("MiniMax H3 text TP requires the MiniMax Qwen3-VL text encoder")
    return clip_model, clip_model.transformer, clip_model.transformer.model


def _validate_tp_heads(language_model, world_size):
    if language_model.config.num_attention_heads % world_size != 0:
        raise ValueError("Qwen attention heads are not divisible by the requested TP size")
    if language_model.config.num_key_value_heads % world_size != 0:
        raise ValueError("Qwen KV heads are not divisible by the requested TP size")


def _prepare_minimax_h3_tp_topology(model, world_size, rank, keep_frontend):
    _, transformer, language_model = _minimax_h3_language_model(model)
    _validate_tp_heads(language_model, world_size)

    for layer in language_model.layers:
        attention = layer.self_attn
        _resize_tp_linear_topology(attention.q_proj, world_size, rank, dim=0)
        _resize_tp_linear_topology(attention.k_proj, world_size, rank, dim=0)
        _resize_tp_linear_topology(attention.v_proj, world_size, rank, dim=0)
        _resize_tp_linear_topology(attention.o_proj, world_size, rank, dim=1)
        attention.num_heads //= world_size
        attention.num_kv_heads //= world_size
        attention.inner_size //= world_size
        if hasattr(attention, "kv_size"):
            attention.kv_size //= world_size

        _resize_tp_linear_topology(layer.mlp.gate_proj, world_size, rank, dim=0)
        _resize_tp_linear_topology(layer.mlp.up_proj, world_size, rank, dim=0)
        _resize_tp_linear_topology(layer.mlp.down_proj, world_size, rank, dim=1)

    if not keep_frontend:
        transformer.visual = None
        language_model.embed_tokens = None


_TP_COLUMN_PROJECTIONS = frozenset(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"))
_TP_ROW_PROJECTIONS = frozenset(("o_proj", "down_proj"))


def _minimax_h3_tp_state_dict(state_dict, world_size, rank, keep_frontend):
    sharded = {}
    for key, value in state_dict.items():
        if not keep_frontend and (key.startswith("visual.") or key.startswith("model.embed_tokens.")):
            continue

        parts = key.split(".")
        projection = parts[-2] if len(parts) >= 2 else ""
        is_language_projection = len(parts) >= 6 and parts[0] == "model" and parts[1] == "layers"
        if not is_language_projection or projection not in _TP_COLUMN_PROJECTIONS | _TP_ROW_PROJECTIONS:
            sharded[key] = value
            continue

        parameter = parts[-1]
        if parameter == "weight":
            dim = 0 if projection in _TP_COLUMN_PROJECTIONS else 1
            sharded[key] = _slice_tp_weight(value, world_size, rank, dim=dim)
        elif parameter == "weight_scale" and projection in _TP_COLUMN_PROJECTIONS:
            # Tensor-wise INT8 stores one scale per output row. Column-parallel
            # projections shard those rows; row-parallel projections retain all
            # output rows and therefore share the complete scale tensor.
            sharded[key] = _slice_tp_weight(value, world_size, rank, dim=0)
        elif parameter == "bias":
            if projection in _TP_COLUMN_PROJECTIONS:
                sharded[key] = _slice_tp_weight(value, world_size, rank, dim=0)
            elif rank == 0:
                sharded[key] = value
        else:
            # Quantization descriptors such as comfy_quant are metadata, not
            # projection matrices. Every rank needs the unchanged descriptor.
            sharded[key] = value
    logging.info("Prepared MiniMax H3 text TP checkpoint shard %d/%d", rank + 1, world_size)
    return sharded


def minimax_h3_tp_reload_model_options(world_size, rank):
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid TP rank {rank} for world size {world_size}")
    keep_frontend = rank == 0
    return {
        "initial_device": torch.device("meta"),
        "pre_load_model_callback": functools.partial(
            _prepare_minimax_h3_tp_topology,
            world_size=world_size,
            rank=rank,
            keep_frontend=keep_frontend,
        ),
        "state_dict_transform": functools.partial(
            _minimax_h3_tp_state_dict,
            world_size=world_size,
            rank=rank,
            keep_frontend=keep_frontend,
        ),
        "assign_state_dict": True,
    }


def _shard_minimax_h3_text_encoder(model, world_size, rank, keep_frontend):
    clip_model, transformer, language_model = _minimax_h3_language_model(model)
    _validate_tp_heads(language_model, world_size)

    for layer in language_model.layers:
        attention = layer.self_attn
        _shard_tp_linear(attention.q_proj, world_size, rank, dim=0)
        _shard_tp_linear(attention.k_proj, world_size, rank, dim=0)
        _shard_tp_linear(attention.v_proj, world_size, rank, dim=0)
        _shard_tp_linear(attention.o_proj, world_size, rank, dim=1)
        attention.num_heads //= world_size
        attention.num_kv_heads //= world_size
        attention.inner_size //= world_size
        if hasattr(attention, "kv_size"):
            attention.kv_size //= world_size

        _shard_tp_linear(layer.mlp.gate_proj, world_size, rank, dim=0)
        _shard_tp_linear(layer.mlp.up_proj, world_size, rank, dim=0)
        _shard_tp_linear(layer.mlp.down_proj, world_size, rank, dim=1)

    if not keep_frontend:
        transformer.visual = None
        language_model.embed_tokens = None
    return clip_model


def configure_minimax_h3_text_tensor_parallel(patchers, devices):
    world_size = len(devices)
    clip_models = []
    for rank, patcher in enumerate(patchers):
        clip_models.append(_shard_minimax_h3_text_encoder(
            patcher.model,
            world_size=world_size,
            rank=rank,
            keep_frontend=rank == 0,
        ))
        patcher.size = 0

    return connect_minimax_h3_text_tensor_parallel(patchers, devices)


def connect_minimax_h3_text_tensor_parallel(patchers, devices):
    clip_models = [_minimax_h3_language_model(patcher.model)[0] for patcher in patchers]
    for patcher in patchers:
        patcher.size = 0
    context = MiniMaxH3TextTensorParallel(clip_models, devices)
    clip_models[0]._minimax_h3_tp_context = context
    patchers[0].set_additional_models("minimax_h3_text_tp", list(patchers[1:]))
    return context


class MiniMaxH3TextTensorParallel:
    def __init__(self, clip_models, devices):
        self.clip_models = tuple(clip_models)
        self.devices = tuple(devices)
        self.world_size = len(devices)

    def _rank_forward(self, rank, embeds, position_ids, visual_pos_masks, deepstack, coordinator):
        device = self.devices[rank]
        try:
            with comfy.model_management.cuda_device_context(device):
                local_embeds = embeds.to(device=device, non_blocking=True)
                local_positions = None if position_ids is None else position_ids.to(device=device, non_blocking=True)
                local_mask = None if visual_pos_masks is None else visual_pos_masks.to(device=device, non_blocking=True)
                local_deepstack = None
                if deepstack is not None:
                    local_deepstack = [item.to(device=device, non_blocking=True) for item in deepstack]
                outputs = self.clip_models[rank].transformer.model(
                    None,
                    embeds=local_embeds,
                    dtype=torch.float32,
                    position_ids=local_positions,
                    visual_pos_masks=local_mask,
                    deepstack_embeds=local_deepstack,
                    tp_context=coordinator,
                    tp_rank=rank,
                )
                comfy.model_management.synchronize()
                return outputs[0].float() if rank == 0 else None
        except Exception:
            coordinator.abort()
            raise

    def encode(self, primary, tokens):
        comfy.model_management.throw_exception_if_processing_interrupted()
        device = self.devices[0]
        with comfy.model_management.cuda_device_context(device):
            embeds, _, _, embeds_info = primary.process_tokens(tokens, device)
            position_ids, visual_pos_masks, deepstack = primary.transformer.build_image_inputs(embeds, embeds_info)
            primary.transformer.last_token_tags = token_tags_from_embeds_info(embeds.shape[1], embeds_info)
            comfy.model_management.synchronize()

        coordinator = comfy.multigpu.TensorParallelCoordinator(list(self.devices))
        pool = comfy.multigpu.MultiGPUThreadPool(list(self.devices[1:]))
        try:
            for rank, rank_device in enumerate(self.devices[1:], start=1):
                pool.submit(rank_device, self._rank_forward, rank, embeds, position_ids, visual_pos_masks, deepstack, coordinator)
            output = self._rank_forward(0, embeds, position_ids, visual_pos_masks, deepstack, coordinator)
            for rank_device in self.devices[1:]:
                _, error = pool.get_result(rank_device)
                if error is not None:
                    raise error
        finally:
            pool.shutdown()

        logging.info("MiniMax H3 text encode completed with TP=%s", self.world_size)
        return output, None


class MiniMaxQwen3VL(Qwen3VL):
    model_type = "qwen3vl_32b"

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image" and embed.get("minimax_video_block", False):
            flatten, grid = process_video_block(embed["data"])
            merged, deepstack = self.visual(flatten.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return super().preprocess_embed(embed, device)

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, embeds_info=[], **kwargs):
        seq = embeds.shape[1] if embeds is not None else input_ids.shape[1]
        self.last_token_tags = token_tags_from_embeds_info(seq, embeds_info)
        return super().forward(input_ids, attention_mask=attention_mask, embeds=embeds,
                               num_tokens=num_tokens, intermediate_output=intermediate_output,
                               final_layer_norm_intermediate=final_layer_norm_intermediate,
                               dtype=dtype, embeds_info=embeds_info, **kwargs)


class TextConnector(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.input_mean = torch.nn.Parameter(torch.empty(in_channels, dtype=dtype, device=device))
        self.input_std = torch.nn.Parameter(torch.empty(in_channels, dtype=dtype, device=device))
        self.proj = operations.Linear(in_channels, out_channels, bias=False, dtype=dtype, device=device)
        self.output_mean = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device))
        self.output_std = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device))
        self.sink_embedding = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device))

    def forward(self, hidden_states):
        # [B, layers, T, C] -> [B, T, layers * C] standardized, projected and rescaled; the first token is replaced by the sink
        x = hidden_states.movedim(1, 2).flatten(2)
        x = (x - comfy.ops.cast_to_input(self.input_mean, x)) / comfy.ops.cast_to_input(self.input_std, x)
        x = self.proj(x)
        x = torch.addcmul(comfy.ops.cast_to_input(self.output_mean, x), x, comfy.ops.cast_to_input(self.output_std, x))
        sink = comfy.ops.cast_to_input(self.sink_embedding, x).expand(x.shape[0], 1, -1)
        return torch.cat([sink, x[:, 1:]], dim=1)


class PH3ImageQwen3VL(MiniMaxQwen3VL):
    model_type = "qwen3vl_4b"

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__(config_dict, dtype, device, operations)
        # float32 and never quantized: the mixed-precision ops of a quantized file would load it at the compute dtype
        self.text_connector = TextConnector(len(PH3_IMAGE_HIDDEN_STATES) * self.model.config.hidden_size, PH3_IMAGE_TEXT_DIM,
                                            dtype=torch.float32, device=device, operations=comfy.ops.manual_cast)

    def forward(self, *args, **kwargs):
        x, hidden_states = super().forward(*args, **kwargs)
        return x, self.text_connector(hidden_states)


class MiniMaxH3ClipModel(comfy.sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_class=MiniMaxQwen3VL, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=None, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=model_class, enable_attention_masks=False,
                         return_attention_masks=False, model_options=model_options)

    def encode(self, tokens):
        tp_context = getattr(self, "_minimax_h3_tp_context", None)
        if tp_context is not None:
            return tp_context.encode(self, tokens)
        return super().encode(tokens)

    def encode_token_weights(self, token_weight_pairs):
        out = super().encode_token_weights(token_weight_pairs)
        tags = getattr(self.transformer, "last_token_tags", None)
        if tags is not None:
            extra = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
            extra["minimax_token_tags"] = tags
            out = (out[0], out[1], extra)
        return out


class PH3ImageClipModel(MiniMaxH3ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, layer=PH3_IMAGE_HIDDEN_STATES, dtype=dtype, model_class=PH3ImageQwen3VL, model_options=model_options)


class MiniMaxH3TEModel(comfy.sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_32b",
                         clip_model=MiniMaxH3ClipModel, model_options=model_options)


class PH3ImageTEModel(comfy.sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b",
                         clip_model=PH3ImageClipModel, model_options=model_options)


class MiniMaxQwenSDTokenizer(Qwen3VLSDTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(MINIMAX_EXTRA_TOKENS)})
        self.inv_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}


class MiniMaxH3Tokenizer(comfy.sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = lambda *a, **kw: MiniMaxQwenSDTokenizer(*a, **kw, embedding_size=5120, embedding_key="qwen3vl_32b")
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3vl_32b", tokenizer=tokenizer)

    @staticmethod
    def _vision_entry(data, video_block=False):
        emb = {"type": "image", "data": data, "original_type": "image"}
        if video_block:
            emb["minimax_video_block"] = True
        return emb

    def tokenize_with_weights(self, text, return_word_ids=False, images=[],
                              minimax_ref_items=None, **kwargs):
        entries = []

        def add_text(s):
            if not s:
                return
            token_batches = getattr(self, self.clip).tokenize_with_weights(
                s,
                return_word_ids=False,
                disable_weights=True,
            )
            if len(token_batches) != 1:
                raise ValueError("MiniMax H3 text segment exceeds the supported prompt length.")
            entries.extend(token_batches[0])

        def add_vision(data, video_block=False):
            entries.append((VISION_START, 1.0))
            entries.append((self._vision_entry(data, video_block), 1.0))
            entries.append((VISION_END, 1.0))

        if minimax_ref_items:
            counters = {"image": 0, "audio": 0, "video": 0}
            for item in minimax_ref_items:
                kind = item["type"]
                counters[kind] += 1
                if kind == "image":
                    add_text("<Picture %d>: " % counters["image"])
                    add_vision(item["data"])
                elif kind == "audio":
                    add_text("<Audio %d>: " % counters["audio"])
                elif kind == "video":
                    frames = item["data"]  # [T, H, W, C], sampled at 2 fps
                    timestamps = item.get("timestamps")
                    if timestamps is None:
                        timestamps = [i / 2.0 for i in range(frames.shape[0])]
                    if frames.shape[0] % 2 == 1:  # repeat-pad to temporal patch of 2
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        timestamps = list(timestamps) + [timestamps[-1]]
                    add_text("<Video %d>: " % counters["video"])
                    for i in range(0, frames.shape[0], 2):
                        block_ts = (timestamps[i] + timestamps[i + 1]) / 2.0
                        add_text("<%.1f seconds>" % block_ts)
                        add_vision(frames[i:i + 2], video_block=True)
        else:
            for i, img in enumerate(images):
                add_text("<Picture %d>: " % (i + 1))
                add_vision(img)

        add_text(text)
        if len(entries) == 0:
            entries.append((151643, 1.0))
        if return_word_ids:
            entries = [t + (0,) for t in entries]
        return {self.clip: [entries]}


class PH3ImageTokenizer(MiniMaxH3Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = lambda *a, **kw: Qwen3VLSDTokenizer(*a, **kw, embedding_size=2560, embedding_key="qwen3vl_4b")
        comfy.sd1_clip.SD1Tokenizer.__init__(self, embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3vl_4b", tokenizer=tokenizer)


def te(dtype_llama=None, llama_quantization_metadata=None, te_model=MiniMaxH3TEModel, **kwargs):
    class MiniMaxH3TEModel_(te_model):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return MiniMaxH3TEModel_
