"""
Unit tests for the sanitizer module.

Tests cover:
- Null byte removal
- Control character stripping
- Line ending normalization
- Unicode handling
- Idempotency
- Preservation of valid content
- Empty but valid input
- Dictionary sanitization
"""

import sys
import unittest

sys.path.insert(0, ".")

from src.validation.sanitizer import (
    Sanitizer,
    sanitize_text,
    sanitize_dict,
    _remove_null_bytes,
    _remove_control_characters,
    _normalize_line_endings,
    _normalize_unicode,
    _collapse_excessive_whitespace,
)


# =============================================================================
# Test: Null Byte Removal
# =============================================================================

class TestNullByteRemoval(unittest.TestCase):
    """Tests for null byte removal."""

    def test_remove_single_null_byte(self):
        """Test removal of a single null byte."""
        text = "hello\x00world"
        result = _remove_null_bytes(text)
        self.assertEqual(result, "helloworld")

    def test_remove_multiple_null_bytes(self):
        """Test removal of multiple null bytes."""
        text = "a\x00b\x00c\x00d"
        result = _remove_null_bytes(text)
        self.assertEqual(result, "abcd")

    def test_remove_consecutive_null_bytes(self):
        """Test removal of consecutive null bytes."""
        text = "hello\x00\x00\x00world"
        result = _remove_null_bytes(text)
        self.assertEqual(result, "helloworld")

    def test_remove_null_at_start(self):
        """Test removal of null byte at start."""
        text = "\x00hello"
        result = _remove_null_bytes(text)
        self.assertEqual(result, "hello")

    def test_remove_null_at_end(self):
        """Test removal of null byte at end."""
        text = "hello\x00"
        result = _remove_null_bytes(text)
        self.assertEqual(result, "hello")

    def test_no_null_bytes(self):
        """Test that text without null bytes is unchanged."""
        text = "hello world"
        result = _remove_null_bytes(text)
        self.assertEqual(result, "hello world")

    def test_empty_string(self):
        """Test empty string."""
        result = _remove_null_bytes("")
        self.assertEqual(result, "")

    def test_only_null_bytes(self):
        """Test string with only null bytes."""
        result = _remove_null_bytes("\x00\x00\x00")
        self.assertEqual(result, "")


# =============================================================================
# Test: Control Character Removal
# =============================================================================

class TestControlCharacterRemoval(unittest.TestCase):
    """Tests for control character removal."""

    def test_remove_bell_character(self):
        """Test removal of bell character (ASCII 7)."""
        text = "hello\x07world"
        result = _remove_control_characters(text)
        self.assertEqual(result, "helloworld")

    def test_remove_backspace(self):
        """Test removal of backspace (ASCII 8)."""
        text = "hello\x08world"
        result = _remove_control_characters(text)
        self.assertEqual(result, "helloworld")

    def test_preserve_tab(self):
        """Test that tab character is preserved."""
        text = "hello\tworld"
        result = _remove_control_characters(text)
        self.assertEqual(result, "hello\tworld")

    def test_preserve_newline(self):
        """Test that newline is preserved."""
        text = "hello\nworld"
        result = _remove_control_characters(text)
        self.assertEqual(result, "hello\nworld")

    def test_preserve_carriage_return(self):
        """Test that carriage return is preserved (for later normalization)."""
        text = "hello\rworld"
        result = _remove_control_characters(text)
        self.assertEqual(result, "hello\rworld")

    def test_remove_escape(self):
        """Test removal of escape character (ASCII 27)."""
        text = "hello\x1bworld"
        result = _remove_control_characters(text)
        self.assertEqual(result, "helloworld")

    def test_remove_delete(self):
        """Test removal of delete character (ASCII 127)."""
        text = "hello\x7fworld"
        result = _remove_control_characters(text)
        self.assertEqual(result, "helloworld")

    def test_remove_form_feed(self):
        """Test removal of form feed (ASCII 12)."""
        text = "hello\x0cworld"
        result = _remove_control_characters(text)
        self.assertEqual(result, "helloworld")

    def test_remove_vertical_tab(self):
        """Test removal of vertical tab (ASCII 11)."""
        text = "hello\x0bworld"
        result = _remove_control_characters(text)
        self.assertEqual(result, "helloworld")

    def test_multiple_control_characters(self):
        """Test removal of multiple different control characters."""
        text = "\x01hello\x02world\x03"
        result = _remove_control_characters(text)
        self.assertEqual(result, "helloworld")

    def test_preserve_normal_text(self):
        """Test that normal text is unchanged."""
        text = "Hello, World! 123"
        result = _remove_control_characters(text)
        self.assertEqual(result, "Hello, World! 123")


# =============================================================================
# Test: Line Ending Normalization
# =============================================================================

class TestLineEndingNormalization(unittest.TestCase):
    """Tests for line ending normalization."""

    def test_normalize_windows_line_ending(self):
        """Test normalization of Windows line endings (\\r\\n -> \\n)."""
        text = "line1\r\nline2\r\nline3"
        result = _normalize_line_endings(text)
        self.assertEqual(result, "line1\nline2\nline3")

    def test_normalize_old_mac_line_ending(self):
        """Test normalization of old Mac line endings (\\r -> \\n)."""
        text = "line1\rline2\rline3"
        result = _normalize_line_endings(text)
        self.assertEqual(result, "line1\nline2\nline3")

    def test_preserve_unix_line_ending(self):
        """Test that Unix line endings are unchanged."""
        text = "line1\nline2\nline3"
        result = _normalize_line_endings(text)
        self.assertEqual(result, "line1\nline2\nline3")

    def test_mixed_line_endings(self):
        """Test normalization of mixed line endings."""
        text = "line1\r\nline2\rline3\nline4"
        result = _normalize_line_endings(text)
        self.assertEqual(result, "line1\nline2\nline3\nline4")

    def test_no_line_endings(self):
        """Test text without line endings."""
        text = "hello world"
        result = _normalize_line_endings(text)
        self.assertEqual(result, "hello world")

    def test_empty_string(self):
        """Test empty string."""
        result = _normalize_line_endings("")
        self.assertEqual(result, "")

    def test_only_line_endings(self):
        """Test string with only line endings."""
        result = _normalize_line_endings("\r\n\r\n")
        self.assertEqual(result, "\n\n")


# =============================================================================
# Test: Unicode Normalization
# =============================================================================

class TestUnicodeNormalization(unittest.TestCase):
    """Tests for Unicode normalization."""

    def test_normalize_composed_character(self):
        """Test normalization of composed character."""
        # U+00E9 (é) vs U+0065 U+0301 (e + combining acute)
        decomposed = "e\u0301"  # e + combining acute accent
        composed = "é"          # precomposed é
        result = _normalize_unicode(decomposed)
        self.assertEqual(result, composed)

    def test_normalize_already_nfc(self):
        """Test that already NFC text is unchanged."""
        text = "café"
        result = _normalize_unicode(text)
        self.assertEqual(result, "café")

    def test_preserve_ascii(self):
        """Test that ASCII text is unchanged."""
        text = "hello world"
        result = _normalize_unicode(text)
        self.assertEqual(result, "hello world")

    def test_normalize_emoji(self):
        """Test that emoji are handled correctly."""
        text = "Hello 😀"
        result = _normalize_unicode(text)
        self.assertEqual(result, "Hello 😀")

    def test_normalize_cjk(self):
        """Test that CJK characters are handled correctly."""
        text = "Hello 世界"
        result = _normalize_unicode(text)
        self.assertEqual(result, "Hello 世界")

    def test_normalize_arabic(self):
        """Test that Arabic text is handled correctly."""
        text = "مرحبا"
        result = _normalize_unicode(text)
        self.assertEqual(result, "مرحبا")

    def test_empty_string(self):
        """Test empty string."""
        result = _normalize_unicode("")
        self.assertEqual(result, "")


# =============================================================================
# Test: Whitespace Collapsing
# =============================================================================

class TestWhitespaceCollapsing(unittest.TestCase):
    """Tests for excessive whitespace collapsing."""

    def test_collapse_excessive_spaces(self):
        """Test collapsing of excessive spaces."""
        text = "a" + " " * 150 + "b"
        result = _collapse_excessive_whitespace(text, max_consecutive=100)
        self.assertEqual(len(result), 102)  # a + 100 spaces + b

    def test_no_collapse_under_limit(self):
        """Test that spaces under limit are not collapsed."""
        text = "a" + " " * 50 + "b"
        result = _collapse_excessive_whitespace(text, max_consecutive=100)
        self.assertEqual(result, text)

    def test_collapse_excessive_newlines(self):
        """Test collapsing of excessive newlines."""
        text = "a" + "\n" * 150 + "b"
        result = _collapse_excessive_whitespace(text, max_consecutive=100)
        self.assertEqual(len(result), 102)

    def test_collapse_excessive_tabs(self):
        """Test collapsing of excessive tabs."""
        text = "a" + "\t" * 150 + "b"
        result = _collapse_excessive_whitespace(text, max_consecutive=100)
        self.assertEqual(len(result), 102)

    def test_mixed_whitespace_not_collapsed(self):
        """Test that mixed whitespace types are not collapsed together."""
        text = "a" + " \n" * 75 + "b"  # Alternating space and newline
        result = _collapse_excessive_whitespace(text, max_consecutive=100)
        self.assertEqual(result, text)

    def test_short_string_unchanged(self):
        """Test that short strings are unchanged."""
        text = "hello world"
        result = _collapse_excessive_whitespace(text, max_consecutive=100)
        self.assertEqual(result, "hello world")

    def test_empty_string(self):
        """Test empty string."""
        result = _collapse_excessive_whitespace("", max_consecutive=100)
        self.assertEqual(result, "")


# =============================================================================
# Test: Sanitizer Class - Text Sanitization
# =============================================================================

class TestSanitizerText(unittest.TestCase):
    """Tests for Sanitizer.sanitize_text method."""

    def setUp(self):
        self.sanitizer = Sanitizer()

    def test_sanitize_clean_text(self):
        """Test that clean text is unchanged."""
        text = "Hello, World!"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, "Hello, World!")

    def test_sanitize_removes_null_bytes(self):
        """Test that null bytes are removed."""
        text = "hello\x00world"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, "helloworld")

    def test_sanitize_removes_control_chars(self):
        """Test that control characters are removed."""
        text = "hello\x07world"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, "helloworld")

    def test_sanitize_preserves_tab(self):
        """Test that tab is preserved."""
        text = "hello\tworld"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, "hello\tworld")

    def test_sanitize_preserves_newline(self):
        """Test that newline is preserved."""
        text = "hello\nworld"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, "hello\nworld")

    def test_sanitize_normalizes_line_endings(self):
        """Test that line endings are normalized."""
        text = "line1\r\nline2"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, "line1\nline2")

    def test_sanitize_normalizes_unicode(self):
        """Test that Unicode is normalized."""
        text = "cafe\u0301"  # cafe + combining acute
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, "café")

    def test_sanitize_empty_string(self):
        """Test sanitization of empty string."""
        result = self.sanitizer.sanitize_text("")
        self.assertEqual(result, "")

    def test_sanitize_whitespace_only(self):
        """Test sanitization of whitespace-only string."""
        result = self.sanitizer.sanitize_text("   \n\t  ")
        self.assertEqual(result, "   \n\t  ")

    def test_sanitize_non_string_returns_empty(self):
        """Test that non-string input returns empty string."""
        result = self.sanitizer.sanitize_text(123)
        self.assertEqual(result, "")

    def test_sanitize_with_trim_whitespace(self):
        """Test sanitization with trim_whitespace option."""
        sanitizer = Sanitizer(trim_whitespace=True)
        result = sanitizer.sanitize_text("  hello  ")
        self.assertEqual(result, "hello")


# =============================================================================
# Test: Idempotency
# =============================================================================

class TestIdempotency(unittest.TestCase):
    """Tests for sanitization idempotency."""

    def setUp(self):
        self.sanitizer = Sanitizer()

    def test_idempotent_clean_text(self):
        """Test idempotency with clean text."""
        text = "Hello, World!"
        result1 = self.sanitizer.sanitize_text(text)
        result2 = self.sanitizer.sanitize_text(result1)
        self.assertEqual(result1, result2)

    def test_idempotent_dirty_text(self):
        """Test idempotency with dirty text."""
        text = "hello\x00\x07world\r\n"
        result1 = self.sanitizer.sanitize_text(text)
        result2 = self.sanitizer.sanitize_text(result1)
        self.assertEqual(result1, result2)

    def test_idempotent_unicode(self):
        """Test idempotency with Unicode text."""
        text = "cafe\u0301 世界 😀"
        result1 = self.sanitizer.sanitize_text(text)
        result2 = self.sanitizer.sanitize_text(result1)
        self.assertEqual(result1, result2)

    def test_idempotent_multiple_passes(self):
        """Test idempotency with multiple passes."""
        text = "hello\x00\x07world\r\ncafe\u0301"
        result = text
        for _ in range(5):
            result = self.sanitizer.sanitize_text(result)
        expected = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, expected)

    def test_idempotent_dict(self):
        """Test idempotency with dictionary."""
        data = {"text": "hello\x00world", "nested": {"value": "test\x07"}}
        result1 = self.sanitizer.sanitize_dict(data)
        result2 = self.sanitizer.sanitize_dict(result1)
        self.assertEqual(result1, result2)


# =============================================================================
# Test: Preservation of Valid Content
# =============================================================================

class TestPreservation(unittest.TestCase):
    """Tests for preservation of valid content."""

    def setUp(self):
        self.sanitizer = Sanitizer()

    def test_preserve_letters(self):
        """Test that letters are preserved."""
        text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_preserve_digits(self):
        """Test that digits are preserved."""
        text = "0123456789"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_preserve_punctuation(self):
        """Test that punctuation is preserved."""
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_preserve_unicode_letters(self):
        """Test that Unicode letters are preserved."""
        text = "Привет мир"  # Russian
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_preserve_cjk_characters(self):
        """Test that CJK characters are preserved."""
        text = "你好世界"  # Chinese
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_preserve_emoji(self):
        """Test that emoji are preserved."""
        text = "😀🎉🚀💯"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_preserve_mathematical_symbols(self):
        """Test that mathematical symbols are preserved."""
        text = "α β γ δ ∑ ∏ √ ∞ ≠ ≤ ≥"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_preserve_currency_symbols(self):
        """Test that currency symbols are preserved."""
        text = "$ € £ ¥ ₹"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_preserve_newlines_in_paragraph(self):
        """Test that newlines in paragraphs are preserved."""
        text = "Line 1\nLine 2\n\nParagraph 2"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_preserve_tabs_for_indentation(self):
        """Test that tabs for indentation are preserved."""
        text = "\tIndented line\n\t\tDouble indented"
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)


# =============================================================================
# Test: Dictionary Sanitization
# =============================================================================

class TestSanitizerDict(unittest.TestCase):
    """Tests for Sanitizer.sanitize_dict method."""

    def setUp(self):
        self.sanitizer = Sanitizer()

    def test_sanitize_simple_dict(self):
        """Test sanitization of simple dictionary."""
        data = {"text": "hello\x00world"}
        result = self.sanitizer.sanitize_dict(data)
        self.assertEqual(result["text"], "helloworld")

    def test_sanitize_multiple_fields(self):
        """Test sanitization of multiple fields."""
        data = {
            "text": "hello\x00world",
            "modality": "text",
            "source": "cli\x07",
        }
        result = self.sanitizer.sanitize_dict(data)
        self.assertEqual(result["text"], "helloworld")
        self.assertEqual(result["modality"], "text")
        self.assertEqual(result["source"], "cli")

    def test_sanitize_nested_dict(self):
        """Test sanitization of nested dictionary."""
        data = {
            "text": "hello",
            "metadata": {
                "key": "value\x00",
                "nested": {
                    "deep": "test\x07",
                }
            }
        }
        result = self.sanitizer.sanitize_dict(data)
        self.assertEqual(result["metadata"]["key"], "value")
        self.assertEqual(result["metadata"]["nested"]["deep"], "test")

    def test_sanitize_list_in_dict(self):
        """Test sanitization of list values in dictionary."""
        data = {
            "items": ["hello\x00", "world\x07"]
        }
        result = self.sanitizer.sanitize_dict(data)
        self.assertEqual(result["items"], ["hello", "world"])

    def test_preserve_non_string_values(self):
        """Test that non-string values are preserved."""
        data = {
            "text": "hello",
            "count": 42,
            "ratio": 3.14,
            "enabled": True,
            "empty": None,
        }
        result = self.sanitizer.sanitize_dict(data)
        self.assertEqual(result["count"], 42)
        self.assertEqual(result["ratio"], 3.14)
        self.assertEqual(result["enabled"], True)
        self.assertIsNone(result["empty"])

    def test_sanitize_empty_dict(self):
        """Test sanitization of empty dictionary."""
        result = self.sanitizer.sanitize_dict({})
        self.assertEqual(result, {})

    def test_sanitize_non_dict_returns_empty(self):
        """Test that non-dict input returns empty dict."""
        result = self.sanitizer.sanitize_dict("not a dict")
        self.assertEqual(result, {})

    def test_sanitize_request_data(self):
        """Test sanitize_request_data method."""
        data = {
            "text": "hello\x00world",
            "modality": "text",
            "metadata": {"key": "value\x07"},
        }
        result = self.sanitizer.sanitize_request_data(data)
        self.assertEqual(result["text"], "helloworld")
        self.assertEqual(result["metadata"]["key"], "value")


# =============================================================================
# Test: Empty and Edge Cases
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def setUp(self):
        self.sanitizer = Sanitizer()

    def test_empty_string(self):
        """Test empty string input."""
        result = self.sanitizer.sanitize_text("")
        self.assertEqual(result, "")

    def test_single_character(self):
        """Test single character input."""
        result = self.sanitizer.sanitize_text("a")
        self.assertEqual(result, "a")

    def test_single_null_byte(self):
        """Test single null byte input."""
        result = self.sanitizer.sanitize_text("\x00")
        self.assertEqual(result, "")

    def test_single_control_char(self):
        """Test single control character input."""
        result = self.sanitizer.sanitize_text("\x07")
        self.assertEqual(result, "")

    def test_single_newline(self):
        """Test single newline input."""
        result = self.sanitizer.sanitize_text("\n")
        self.assertEqual(result, "\n")

    def test_very_long_string(self):
        """Test very long string input."""
        text = "x" * 100000
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, text)

    def test_all_control_characters(self):
        """Test string with all ASCII control characters."""
        text = "".join(chr(i) for i in range(32))
        result = self.sanitizer.sanitize_text(text)
        # Only tab (9), newline (10), carriage return (13) should remain
        # CR normalized to LF
        self.assertEqual(result, "\t\n\n")

    def test_unicode_control_characters(self):
        """Test Unicode control character removal."""
        text = "hello\u0085world"  # NEL (Next Line)
        result = self.sanitizer.sanitize_text(text)
        self.assertEqual(result, "helloworld")


# =============================================================================
# Test: Convenience Functions
# =============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def test_sanitize_text_function(self):
        """Test sanitize_text convenience function."""
        result = sanitize_text("hello\x00world")
        self.assertEqual(result, "helloworld")

    def test_sanitize_dict_function(self):
        """Test sanitize_dict convenience function."""
        result = sanitize_dict({"text": "hello\x00world"})
        self.assertEqual(result["text"], "helloworld")


# =============================================================================
# Test: Sanitizer Options
# =============================================================================

class TestSanitizerOptions(unittest.TestCase):
    """Tests for sanitizer configuration options."""

    def test_trim_whitespace_enabled(self):
        """Test with trim_whitespace enabled."""
        sanitizer = Sanitizer(trim_whitespace=True)
        result = sanitizer.sanitize_text("  hello world  ")
        self.assertEqual(result, "hello world")

    def test_trim_whitespace_disabled(self):
        """Test with trim_whitespace disabled (default)."""
        sanitizer = Sanitizer(trim_whitespace=False)
        result = sanitizer.sanitize_text("  hello world  ")
        self.assertEqual(result, "  hello world  ")

    def test_collapse_whitespace_enabled(self):
        """Test with collapse_whitespace enabled (default)."""
        sanitizer = Sanitizer(collapse_whitespace=True, max_consecutive_whitespace=10)
        text = "a" + " " * 20 + "b"
        result = sanitizer.sanitize_text(text)
        self.assertEqual(len(result), 12)  # a + 10 spaces + b

    def test_collapse_whitespace_disabled(self):
        """Test with collapse_whitespace disabled."""
        sanitizer = Sanitizer(collapse_whitespace=False)
        text = "a" + " " * 200 + "b"
        result = sanitizer.sanitize_text(text)
        self.assertEqual(len(result), 202)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
