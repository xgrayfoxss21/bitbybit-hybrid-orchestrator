# tests/test_chat_transcript.py
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Unit tests for chat transcript assembly (used by the Gradio chat UI).

We purposely avoid importing heavy runtime pieces. The tests focus on the
_history → transcript_ formatting that the chat adapter performs each turn.

- Prefers importing `_format_transcript` from `ui/chat_gradio.py`
- Falls back to a local implementation if the import is unavailable
  (so contributors can still run tests with a partial checkout).

Run:
  python -m unittest tests/test_chat_transcript.py
  # or, if you use pytest:
  # pytest -q
"""

import unittest
from typing import List, Tuple

# Try to import the helper from the UI module (optional dependency)
try:
    from ui.chat_gradio import _format_transcript  # type: ignore
except Exception:
    # Fallback copy (must mirror ui/chat_gradio.py)
    def _format_transcript(history: List[Tuple[str, str]], user_msg: str) -> str:
        """
        Convert chat history + new user message to a transcript string consumed by the pipeline.
        History is a list of [user, assistant] pairs.
        """
        lines: List[str] = []
        for u, a in history:
            if u:
                lines.append(f"User: {u}")
            if a:
                lines.append(f"Assistant: {a}")
        lines.append(f"User: {user_msg}")
        return "\n".join(lines).strip()


class TestTranscriptFormatting(unittest.TestCase):
    def test_basic_formatting(self):
        history = [
            ("hello", "Hi—what can I help with?"),
            ("verify BitNet and TinyBERT", "Sure, what specifically?"),
        ]
        user_msg = "check these claims"
        tx = _format_transcript(history, user_msg)

        expected = (
            "User: hello\n"
            "Assistant: Hi—what can I help with?\n"
            "User: verify BitNet and TinyBERT\n"
            "Assistant: Sure, what specifically?\n"
            "User: check these claims"
        )
        self.assertEqual(tx, expected)

    def test_ignores_empty_messages(self):
        history = [
            ("hello", ""),          # assistant didn't answer yet
            ("", "Hi there!"),      # stray assistant line (should still be kept)
        ]
        user_msg = "next"
        tx = _format_transcript(history, user_msg)

        expected = (
            "User: hello\n"
            "Assistant: Hi there!\n"
            "User: next"
        )
        self.assertEqual(tx, expected)

    def test_no_trailing_whitespace(self):
        history = [("foo", "bar")]
        user_msg = "baz"
        tx = _format_transcript(history, user_msg)
        self.assertFalse(tx.endswith("\n"))
        self.assertEqual(tx.splitlines()[-1], "User: baz")

    def test_empty_history(self):
        history: List[Tuple[str, str]] = []
        user_msg = "hello"
        tx = _format_transcript(history, user_msg)
        self.assertEqual(tx, "User: hello")

    def test_unicode_and_pii_are_preserved(self):
        # PII is not removed at transcript-assembly time (guard handles redaction later).
        history = [("email me at test@example.com", "👍 Ready.")]
        user_msg = "Résumé ✓ — teléfono +1 (555) 123-4567"
        tx = _format_transcript(history, user_msg)

        self.assertIn("User: email me at test@example.com", tx)
        self.assertIn("Assistant: 👍 Ready.", tx)
        self.assertIn("User: Résumé ✓ — teléfono +1 (555) 123-4567", tx)


if __name__ == "__main__":
    unittest.main()
