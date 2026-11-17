import tempfile
import unittest
from pathlib import Path

from comfy.prompt_history import PromptHistoryStore


class TestPromptHistoryStore(unittest.TestCase):
    def test_history_survives_reopen_in_completion_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "history.sqlite3")
            first = PromptHistoryStore(path)
            first.save("prompt-b", {"prompt": [1, "prompt-b"], "outputs": {}})
            first.save("prompt-a", {"prompt": [2, "prompt-a"], "outputs": {"3": {"text": ["中文"]}}})
            first.close()

            second = PromptHistoryStore(path)
            self.assertEqual(list(second.load()), ["prompt-b", "prompt-a"])
            self.assertEqual(second.load()["prompt-a"]["outputs"]["3"]["text"], ["中文"])
            second.close()

    def test_delete_and_clear_are_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "history.sqlite3")
            store = PromptHistoryStore(path)
            store.save("one", {"outputs": {}})
            store.save("two", {"outputs": {}})
            store.delete("one")
            self.assertEqual(list(store.load()), ["two"])
            store.clear()
            self.assertEqual(store.load(), {})
            store.close()

    def test_updating_an_item_does_not_change_completion_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "history.sqlite3")
            store = PromptHistoryStore(path)
            store.save("one", {"status": {"completed": False}})
            store.save("two", {"status": {"completed": True}})
            store.save("one", {"status": {"completed": True}})
            self.assertEqual(list(store.load()), ["one", "two"])
            self.assertTrue(store.load()["one"]["status"]["completed"])
            store.close()


if __name__ == "__main__":
    unittest.main()
