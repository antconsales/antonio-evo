"""
Sanitizer - Sanitizes validated input data.

This module operates ONLY on already validated data.
It removes potentially dangerous or non-printable content
without altering semantic meaning.

Design principles:
- Operates only on validated data
- Deterministic (same input -> same output)
- Idempotent (sanitize(sanitize(x)) == sanitize(x))
- Preserves original meaning
- No validation (that's done before sanitization)
- No policy decisions
- No logging
- No side effects
"""

import unicodedata
from typing import Any, Dict, Optional


# =============================================================================
# Constants
# =============================================================================

# Control characters to remove (ASCII 0-31 and 127)
# Exceptions: tab (9), newline (10), carriage return (13)
CONTROL_CHARS_TO_REMOVE = frozenset(
    chr(i) for i in range(32) if i not in (9, 10, 13)
) | frozenset([chr(127)])

# Line ending variants to normalize
LINE_ENDINGS = {
    "\r\n": "\n",  # Windows -> Unix
    "\r": "\n",    # Old Mac -> Unix
}

# Unicode categories for control characters
# Cc = Control, Cf = Format (some should be kept)
UNICODE_CONTROL_CATEGORIES = frozenset(["Cc"])


# =============================================================================
# Helper Functions
# =============================================================================

def _remove_null_bytes(text: str) -> str:
    """
    Remove null bytes from text.

    Null bytes can cause issues with C-based libraries and databases.
    This is a security measure.

    Args:
        text: Input text

    Returns:
        Text with null bytes removed
    """
    return text.replace("\x00", "")


def _remove_control_characters(text: str) -> str:
    """
    Remove control characters from text.

    Preserves:
    - Tab (\\t, ASCII 9)
    - Newline (\\n, ASCII 10)
    - Carriage return (\\r, ASCII 13) - will be normalized later

    Removes:
    - All other ASCII control characters (0-31, 127)
    - Unicode control characters (category Cc) except allowed ones

    Args:
        text: Input text

    Returns:
        Text with control characters removed
    """
    result = []
    for char in text:
        # Allow tab, newline, carriage return
        if char in ("\t", "\n", "\r"):
            result.append(char)
            continue

        # Check if it's a control character
        if char in CONTROL_CHARS_TO_REMOVE:
            continue

        # Check Unicode category for non-ASCII
        if ord(char) > 127:
            category = unicodedata.category(char)
            if category in UNICODE_CONTROL_CATEGORIES:
                continue

        result.append(char)

    return "".join(result)


def _normalize_line_endings(text: str) -> str:
    """
    Normalize line endings to Unix style (\\n).

    Converts:
    - \\r\\n (Windows) -> \\n
    - \\r (old Mac) -> \\n

    Args:
        text: Input text

    Returns:
        Text with normalized line endings
    """
    # Replace Windows line endings first (order matters)
    text = text.replace("\r\n", "\n")
    # Then replace any remaining carriage returns
    text = text.replace("\r", "\n")
    return text


def _normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form.

    NFC (Canonical Decomposition, followed by Canonical Composition)
    ensures consistent representation of characters that can be
    represented in multiple ways.

    Example: 'é' can be U+00E9 or U+0065 U+0301
    NFC normalizes to the composed form U+00E9.

    Args:
        text: Input text

    Returns:
        NFC-normalized text
    """
    return unicodedata.normalize("NFC", text)


def _collapse_excessive_whitespace(text: str, max_consecutive: int = 100) -> str:
    """
    Collapse excessive consecutive whitespace.

    This prevents DoS via whitespace padding while preserving
    intentional formatting like paragraphs.

    Only collapses runs of identical whitespace characters longer
    than max_consecutive.

    Args:
        text: Input text
        max_consecutive: Maximum allowed consecutive identical whitespace

    Returns:
        Text with excessive whitespace collapsed
    """
    if len(text) < max_consecutive:
        return text

    result = []
    consecutive_count = 0
    last_char = None

    for char in text:
        if char in (" ", "\t", "\n") and char == last_char:
            consecutive_count += 1
            if consecutive_count <= max_consecutive:
                result.append(char)
        else:
            consecutive_count = 1
            result.append(char)

        last_char = char

    return "".join(result)


# =============================================================================
# Main Sanitizer Class
# =============================================================================

class Sanitizer:
    """
    Sanitizes validated input data.

    Usage:
        sanitizer = Sanitizer()
        clean_text = sanitizer.sanitize_text("hello\\x00world")
        clean_data = sanitizer.sanitize_dict(validated_data)

    The sanitizer:
    - Removes null bytes
    - Removes control characters (except tab, newline)
    - Normalizes line endings to Unix style
    - Normalizes Unicode to NFC
    - Optionally trims whitespace
    - Is deterministic and idempotent
    """

    def __init__(
        self,
        trim_whitespace: bool = False,
        collapse_whitespace: bool = True,
        max_consecutive_whitespace: int = 100,
    ):
        """
        Initialize sanitizer with options.

        Args:
            trim_whitespace: If True, strip leading/trailing whitespace from text
            collapse_whitespace: If True, collapse excessive consecutive whitespace
            max_consecutive_whitespace: Max allowed consecutive identical whitespace chars
        """
        self.trim_whitespace = trim_whitespace
        self.collapse_whitespace = collapse_whitespace
        self.max_consecutive_whitespace = max_consecutive_whitespace

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize a text string.

        Operations performed (in order):
        1. Remove null bytes
        2. Remove control characters (except tab, newline, CR)
        3. Normalize line endings to Unix style
        4. Normalize Unicode to NFC
        5. Optionally collapse excessive whitespace
        6. Optionally trim leading/trailing whitespace

        Args:
            text: Input text (must be a string)

        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            # Should not happen if validation was done first
            # Return empty string to be safe
            return ""

        # Step 1: Remove null bytes
        text = _remove_null_bytes(text)

        # Step 2: Remove control characters
        text = _remove_control_characters(text)

        # Step 3: Normalize line endings
        text = _normalize_line_endings(text)

        # Step 4: Normalize Unicode
        text = _normalize_unicode(text)

        # Step 5: Collapse excessive whitespace
        if self.collapse_whitespace:
            text = _collapse_excessive_whitespace(text, self.max_consecutive_whitespace)

        # Step 6: Trim whitespace
        if self.trim_whitespace:
            text = text.strip()

        return text

    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a dictionary of validated data.

        Sanitizes string values in the dictionary.
        Does not modify the dictionary structure.

        Args:
            data: Dictionary with validated data

        Returns:
            Dictionary with sanitized string values
        """
        if not isinstance(data, dict):
            return {}

        result = {}
        for key, value in data.items():
            result[key] = self._sanitize_value(value)
        return result

    def _sanitize_value(self, value: Any) -> Any:
        """
        Sanitize a single value based on its type.

        Args:
            value: Value to sanitize

        Returns:
            Sanitized value (same type as input)
        """
        if value is None:
            return None

        if isinstance(value, str):
            return self.sanitize_text(value)

        if isinstance(value, dict):
            return self.sanitize_dict(value)

        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]

        # Numbers, booleans, etc. pass through unchanged
        return value

    def sanitize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize validated request data.

        Specifically handles the Request schema fields:
        - text: Sanitized
        - modality: Sanitized
        - task_type: Sanitized
        - quality: Sanitized
        - source: Sanitized
        - audio_path: Sanitized (path strings)
        - image_path: Sanitized (path strings)
        - metadata: Recursively sanitized

        Args:
            data: Validated request data

        Returns:
            Sanitized request data
        """
        return self.sanitize_dict(data)


# =============================================================================
# Convenience Functions
# =============================================================================

def sanitize_text(text: str) -> str:
    """
    Convenience function to sanitize text with default settings.

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    return Sanitizer().sanitize_text(text)


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to sanitize a dictionary with default settings.

    Args:
        data: Input dictionary

    Returns:
        Sanitized dictionary
    """
    return Sanitizer().sanitize_dict(data)
