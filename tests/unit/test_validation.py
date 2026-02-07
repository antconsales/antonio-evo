"""
Unit tests for the validation module.

Tests cover:
- Valid input
- Missing fields
- Wrong types
- Oversized input
- Invalid modality
- Edge cases (empty string, nulls, null bytes)
"""

import sys
import unittest

sys.path.insert(0, ".")

from src.validation import (
    Validator,
    ValidationResult,
    ValidationError,
    FieldError,
    RequestSchema,
    ALLOWED_MODALITIES,
    ALLOWED_TASK_TYPES,
    ALLOWED_QUALITY_LEVELS,
    MAX_TEXT_LENGTH,
    MAX_METADATA_SIZE,
)
from src.validation.schema import (
    ValidationErrorType,
    FieldSchema,
)


# =============================================================================
# Test: Schema Constants
# =============================================================================

class TestSchemaConstants(unittest.TestCase):
    """Tests for schema constants and allowed values."""

    def test_allowed_modalities_not_empty(self):
        """Test that modalities set is not empty."""
        self.assertGreater(len(ALLOWED_MODALITIES), 0)

    def test_allowed_modalities_contains_text(self):
        """Test that 'text' is an allowed modality."""
        self.assertIn("text", ALLOWED_MODALITIES)

    def test_allowed_modalities_contains_audio(self):
        """Test that audio modalities are allowed."""
        self.assertIn("audio_input", ALLOWED_MODALITIES)
        self.assertIn("audio_output", ALLOWED_MODALITIES)

    def test_allowed_task_types_not_empty(self):
        """Test that task types set is not empty."""
        self.assertGreater(len(ALLOWED_TASK_TYPES), 0)

    def test_allowed_task_types_contains_reason(self):
        """Test that 'reason' is an allowed task type."""
        self.assertIn("reason", ALLOWED_TASK_TYPES)

    def test_allowed_quality_levels(self):
        """Test that quality levels are defined."""
        self.assertIn("low", ALLOWED_QUALITY_LEVELS)
        self.assertIn("standard", ALLOWED_QUALITY_LEVELS)
        self.assertIn("high", ALLOWED_QUALITY_LEVELS)

    def test_max_text_length_reasonable(self):
        """Test that max text length is reasonable."""
        self.assertGreater(MAX_TEXT_LENGTH, 100)
        self.assertLess(MAX_TEXT_LENGTH, 1000000)

    def test_max_metadata_size_reasonable(self):
        """Test that max metadata size is reasonable."""
        self.assertGreater(MAX_METADATA_SIZE, 100)
        self.assertLess(MAX_METADATA_SIZE, 100000)


# =============================================================================
# Test: RequestSchema
# =============================================================================

class TestRequestSchema(unittest.TestCase):
    """Tests for RequestSchema class."""

    def test_has_text_field(self):
        """Test that schema has text field."""
        field = RequestSchema.get_field("text")
        self.assertIsNotNone(field)
        self.assertEqual(field.field_type, str)

    def test_has_modality_field(self):
        """Test that schema has modality field."""
        field = RequestSchema.get_field("modality")
        self.assertIsNotNone(field)
        self.assertEqual(field.field_type, str)
        self.assertIsNotNone(field.allowed_values)

    def test_has_metadata_field(self):
        """Test that schema has metadata field."""
        field = RequestSchema.get_field("metadata")
        self.assertIsNotNone(field)
        self.assertEqual(field.field_type, dict)

    def test_get_all_fields(self):
        """Test that get_all_fields returns field names."""
        fields = RequestSchema.get_all_fields()
        self.assertIn("text", fields)
        self.assertIn("modality", fields)
        self.assertIn("metadata", fields)

    def test_get_defaults(self):
        """Test that get_defaults returns default values."""
        defaults = RequestSchema.get_defaults()
        self.assertEqual(defaults.get("modality"), "text")
        self.assertEqual(defaults.get("task_type"), "reason")
        self.assertEqual(defaults.get("quality"), "standard")

    def test_get_nonexistent_field(self):
        """Test that get_field returns None for unknown field."""
        field = RequestSchema.get_field("nonexistent")
        self.assertIsNone(field)


# =============================================================================
# Test: ValidationResult
# =============================================================================

class TestValidationResult(unittest.TestCase):
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid result structure."""
        result = ValidationResult(valid=True, data={"text": "hello"})
        self.assertTrue(result.valid)
        self.assertEqual(result.data["text"], "hello")
        self.assertEqual(len(result.errors), 0)

    def test_invalid_result(self):
        """Test invalid result structure."""
        errors = [FieldError(
            field="text",
            error_type=ValidationErrorType.TOO_LONG,
            message="Too long",
        )]
        result = ValidationResult(valid=False, errors=errors)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 1)

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ValidationResult(valid=True)
        d = result.to_dict()
        self.assertEqual(d["valid"], True)
        self.assertEqual(d["error_count"], 0)


# =============================================================================
# Test: FieldError
# =============================================================================

class TestFieldError(unittest.TestCase):
    """Tests for FieldError dataclass."""

    def test_field_error_creation(self):
        """Test creating a field error."""
        error = FieldError(
            field="text",
            error_type=ValidationErrorType.TOO_LONG,
            message="Text too long",
            value="length=15000",
            constraint=10000,
        )
        self.assertEqual(error.field, "text")
        self.assertEqual(error.error_type, ValidationErrorType.TOO_LONG)

    def test_to_dict(self):
        """Test serialization to dict."""
        error = FieldError(
            field="text",
            error_type=ValidationErrorType.TOO_LONG,
            message="Text too long",
        )
        d = error.to_dict()
        self.assertEqual(d["field"], "text")
        self.assertEqual(d["error_type"], "too_long")
        self.assertEqual(d["message"], "Text too long")

    def test_long_value_truncated(self):
        """Test that long values are truncated in to_dict."""
        long_value = "x" * 200
        error = FieldError(
            field="text",
            error_type=ValidationErrorType.INVALID_VALUE,
            message="Invalid",
            value=long_value,
        )
        d = error.to_dict()
        self.assertLess(len(d["value"]), 150)
        self.assertTrue(d["value"].endswith("..."))


# =============================================================================
# Test: ValidationError Exception
# =============================================================================

class TestValidationError(unittest.TestCase):
    """Tests for ValidationError exception."""

    def test_exception_creation(self):
        """Test creating validation error exception."""
        errors = [FieldError(
            field="text",
            error_type=ValidationErrorType.TOO_LONG,
            message="Text too long",
        )]
        exc = ValidationError(errors)
        self.assertEqual(len(exc.errors), 1)
        self.assertIn("1 error", str(exc))

    def test_exception_to_dict(self):
        """Test exception serialization."""
        errors = [FieldError(
            field="text",
            error_type=ValidationErrorType.TOO_LONG,
            message="Text too long",
        )]
        exc = ValidationError(errors)
        d = exc.to_dict()
        self.assertFalse(d["valid"])
        self.assertEqual(d["error_count"], 1)


# =============================================================================
# Test: Validator - Valid Input
# =============================================================================

class TestValidatorValidInput(unittest.TestCase):
    """Tests for valid input scenarios."""

    def setUp(self):
        self.validator = Validator()

    def test_simple_text(self):
        """Test validation of simple text input."""
        result = self.validator.validate({"text": "Hello world"})
        self.assertTrue(result.valid)
        self.assertEqual(result.data["text"], "Hello world")

    def test_string_input_converted(self):
        """Test that string input is converted to dict with text field."""
        result = self.validator.validate("Hello world")
        self.assertTrue(result.valid)
        self.assertEqual(result.data["text"], "Hello world")

    def test_all_fields(self):
        """Test validation with all fields provided."""
        data = {
            "text": "Hello",
            "modality": "text",
            "task_type": "reason",
            "quality": "standard",
            "source": "cli",
            "metadata": {"key": "value"},
        }
        result = self.validator.validate(data)
        self.assertTrue(result.valid)
        self.assertEqual(result.data["text"], "Hello")
        self.assertEqual(result.data["modality"], "text")

    def test_minimal_input(self):
        """Test validation with minimal input (defaults applied)."""
        result = self.validator.validate({})
        self.assertTrue(result.valid)
        # Defaults should be applied
        self.assertEqual(result.data.get("modality"), "text")
        self.assertEqual(result.data.get("task_type"), "reason")

    def test_empty_text_allowed(self):
        """Test that empty text is allowed."""
        result = self.validator.validate({"text": ""})
        self.assertTrue(result.valid)
        self.assertEqual(result.data["text"], "")

    def test_all_modalities_valid(self):
        """Test that all allowed modalities are valid."""
        for modality in ALLOWED_MODALITIES:
            result = self.validator.validate({"modality": modality})
            self.assertTrue(result.valid, f"Modality '{modality}' should be valid")

    def test_all_task_types_valid(self):
        """Test that all allowed task types are valid."""
        for task_type in ALLOWED_TASK_TYPES:
            result = self.validator.validate({"task_type": task_type})
            self.assertTrue(result.valid, f"Task type '{task_type}' should be valid")

    def test_all_quality_levels_valid(self):
        """Test that all allowed quality levels are valid."""
        for quality in ALLOWED_QUALITY_LEVELS:
            result = self.validator.validate({"quality": quality})
            self.assertTrue(result.valid, f"Quality '{quality}' should be valid")

    def test_metadata_dict(self):
        """Test that metadata dict is valid."""
        result = self.validator.validate({
            "metadata": {"key": "value", "number": 42, "nested": {"a": 1}}
        })
        self.assertTrue(result.valid)

    def test_audio_path(self):
        """Test that audio_path is valid."""
        result = self.validator.validate({
            "audio_path": "/path/to/audio.wav"
        })
        self.assertTrue(result.valid)

    def test_image_path(self):
        """Test that image_path is valid."""
        result = self.validator.validate({
            "image_path": "/path/to/image.png"
        })
        self.assertTrue(result.valid)


# =============================================================================
# Test: Validator - Missing Fields
# =============================================================================

class TestValidatorMissingFields(unittest.TestCase):
    """Tests for missing field handling."""

    def setUp(self):
        self.validator = Validator()

    def test_missing_optional_fields_ok(self):
        """Test that missing optional fields are OK."""
        result = self.validator.validate({"text": "Hello"})
        self.assertTrue(result.valid)

    def test_defaults_applied_for_missing_fields(self):
        """Test that defaults are applied for missing fields."""
        result = self.validator.validate({})
        self.assertTrue(result.valid)
        self.assertEqual(result.data.get("modality"), "text")
        self.assertEqual(result.data.get("task_type"), "reason")
        self.assertEqual(result.data.get("quality"), "standard")
        self.assertEqual(result.data.get("source"), "unknown")


# =============================================================================
# Test: Validator - Wrong Types
# =============================================================================

class TestValidatorWrongTypes(unittest.TestCase):
    """Tests for type validation errors."""

    def setUp(self):
        self.validator = Validator()

    def test_text_not_string(self):
        """Test that non-string text is rejected."""
        result = self.validator.validate({"text": 123})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_A_STRING)

    def test_text_list_rejected(self):
        """Test that list as text is rejected."""
        result = self.validator.validate({"text": ["hello", "world"]})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_A_STRING)

    def test_text_dict_rejected(self):
        """Test that dict as text is rejected."""
        result = self.validator.validate({"text": {"key": "value"}})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_A_STRING)

    def test_modality_not_string(self):
        """Test that non-string modality is rejected."""
        result = self.validator.validate({"modality": 123})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_A_STRING)

    def test_metadata_not_dict(self):
        """Test that non-dict metadata is rejected."""
        result = self.validator.validate({"metadata": "not a dict"})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_A_DICT)

    def test_metadata_list_rejected(self):
        """Test that list as metadata is rejected."""
        result = self.validator.validate({"metadata": [1, 2, 3]})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_A_DICT)

    def test_input_not_dict(self):
        """Test that non-dict input is rejected (except string)."""
        result = self.validator.validate(123)
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_A_DICT)

    def test_input_list_rejected(self):
        """Test that list input is rejected."""
        result = self.validator.validate([{"text": "hello"}])
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_A_DICT)


# =============================================================================
# Test: Validator - Oversized Input
# =============================================================================

class TestValidatorOversizedInput(unittest.TestCase):
    """Tests for oversized input handling."""

    def setUp(self):
        self.validator = Validator()

    def test_text_too_long(self):
        """Test that oversized text is rejected."""
        long_text = "x" * (MAX_TEXT_LENGTH + 1)
        result = self.validator.validate({"text": long_text})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.TOO_LONG)
        self.assertEqual(result.errors[0].field, "text")

    def test_text_at_max_length_ok(self):
        """Test that text at max length is OK."""
        max_text = "x" * MAX_TEXT_LENGTH
        result = self.validator.validate({"text": max_text})
        self.assertTrue(result.valid)

    def test_text_just_over_max_rejected(self):
        """Test that text just over max is rejected."""
        over_text = "x" * (MAX_TEXT_LENGTH + 1)
        result = self.validator.validate({"text": over_text})
        self.assertFalse(result.valid)

    def test_metadata_too_large(self):
        """Test that oversized metadata is rejected."""
        large_metadata = {"data": "x" * (MAX_METADATA_SIZE + 100)}
        result = self.validator.validate({"metadata": large_metadata})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.METADATA_TOO_LARGE)

    def test_source_too_long(self):
        """Test that oversized source is rejected."""
        result = self.validator.validate({"source": "x" * 100})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.TOO_LONG)


# =============================================================================
# Test: Validator - Invalid Values
# =============================================================================

class TestValidatorInvalidValues(unittest.TestCase):
    """Tests for invalid value handling."""

    def setUp(self):
        self.validator = Validator()

    def test_invalid_modality(self):
        """Test that invalid modality is rejected."""
        result = self.validator.validate({"modality": "invalid_modality"})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_IN_ALLOWED_VALUES)
        self.assertEqual(result.errors[0].field, "modality")

    def test_invalid_task_type(self):
        """Test that invalid task type is rejected."""
        result = self.validator.validate({"task_type": "invalid_task"})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_IN_ALLOWED_VALUES)

    def test_invalid_quality(self):
        """Test that invalid quality is rejected."""
        result = self.validator.validate({"quality": "invalid_quality"})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_IN_ALLOWED_VALUES)

    def test_invalid_source(self):
        """Test that invalid source is rejected."""
        result = self.validator.validate({"source": "invalid_source"})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NOT_IN_ALLOWED_VALUES)

    def test_empty_modality_rejected(self):
        """Test that empty modality is rejected."""
        result = self.validator.validate({"modality": ""})
        self.assertFalse(result.valid)
        # Could be EMPTY_VALUE or NOT_IN_ALLOWED_VALUES
        self.assertIn(result.errors[0].error_type, [
            ValidationErrorType.EMPTY_VALUE,
            ValidationErrorType.NOT_IN_ALLOWED_VALUES,
        ])


# =============================================================================
# Test: Validator - Edge Cases
# =============================================================================

class TestValidatorEdgeCases(unittest.TestCase):
    """Tests for edge cases."""

    def setUp(self):
        self.validator = Validator()

    def test_null_input(self):
        """Test that null input is rejected."""
        result = self.validator.validate(None)
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.NULL_VALUE)

    def test_empty_dict(self):
        """Test that empty dict is valid (defaults applied)."""
        result = self.validator.validate({})
        self.assertTrue(result.valid)

    def test_empty_string_input(self):
        """Test that empty string input is valid."""
        result = self.validator.validate("")
        self.assertTrue(result.valid)
        self.assertEqual(result.data["text"], "")

    def test_null_bytes_in_text(self):
        """Test that null bytes in text are detected."""
        result = self.validator.validate({"text": "hello\x00world"})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.CONTAINS_NULL_BYTES)

    def test_null_bytes_in_modality(self):
        """Test that null bytes in modality are detected."""
        result = self.validator.validate({"modality": "text\x00extra"})
        self.assertFalse(result.valid)
        # Should fail for null bytes or invalid value
        error_types = [e.error_type for e in result.errors]
        self.assertTrue(
            ValidationErrorType.CONTAINS_NULL_BYTES in error_types or
            ValidationErrorType.NOT_IN_ALLOWED_VALUES in error_types
        )

    def test_whitespace_only_text(self):
        """Test that whitespace-only text is valid."""
        result = self.validator.validate({"text": "   \n\t  "})
        self.assertTrue(result.valid)

    def test_unicode_text(self):
        """Test that unicode text is valid."""
        result = self.validator.validate({"text": "Hello 世界 Привет"})
        self.assertTrue(result.valid)

    def test_emoji_text(self):
        """Test that emoji text is valid."""
        result = self.validator.validate({"text": "Hello 😀🎉"})
        self.assertTrue(result.valid)

    def test_metadata_not_serializable(self):
        """Test that non-serializable metadata is rejected."""
        # Create a non-serializable object
        class NotSerializable:
            pass

        result = self.validator.validate({"metadata": {"obj": NotSerializable()}})
        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].error_type, ValidationErrorType.METADATA_NOT_SERIALIZABLE)

    def test_unknown_fields_ignored(self):
        """Test that unknown fields are ignored (not error)."""
        result = self.validator.validate({
            "text": "hello",
            "unknown_field": "value",
            "another_unknown": 123,
        })
        self.assertTrue(result.valid)
        # Unknown fields should not be in validated data
        self.assertNotIn("unknown_field", result.data)

    def test_null_text_uses_default(self):
        """Test that null text uses default empty string."""
        result = self.validator.validate({"text": None})
        self.assertTrue(result.valid)
        # Default empty string should be used
        self.assertEqual(result.data.get("text", ""), "")

    def test_boolean_values_in_metadata(self):
        """Test that boolean values in metadata are valid."""
        result = self.validator.validate({
            "metadata": {"flag": True, "other": False}
        })
        self.assertTrue(result.valid)

    def test_nested_dict_in_metadata(self):
        """Test that nested dicts in metadata are valid."""
        result = self.validator.validate({
            "metadata": {
                "level1": {
                    "level2": {
                        "level3": "value"
                    }
                }
            }
        })
        self.assertTrue(result.valid)


# =============================================================================
# Test: Validator - Convenience Methods
# =============================================================================

class TestValidatorConvenienceMethods(unittest.TestCase):
    """Tests for validator convenience methods."""

    def setUp(self):
        self.validator = Validator()

    def test_validate_text(self):
        """Test validate_text convenience method."""
        result = self.validator.validate_text("Hello world")
        self.assertTrue(result.valid)
        self.assertEqual(result.data["text"], "Hello world")

    def test_validate_text_invalid(self):
        """Test validate_text with invalid input."""
        result = self.validator.validate_text(123)
        self.assertFalse(result.valid)

    def test_validate_or_raise_valid(self):
        """Test validate_or_raise with valid input."""
        data = self.validator.validate_or_raise({"text": "Hello"})
        self.assertEqual(data["text"], "Hello")

    def test_validate_or_raise_invalid(self):
        """Test validate_or_raise with invalid input raises."""
        with self.assertRaises(ValidationError) as context:
            self.validator.validate_or_raise({"text": 123})

        self.assertEqual(len(context.exception.errors), 1)


# =============================================================================
# Test: Multiple Errors
# =============================================================================

class TestValidatorMultipleErrors(unittest.TestCase):
    """Tests for multiple error handling."""

    def setUp(self):
        self.validator = Validator()

    def test_multiple_errors_collected(self):
        """Test that multiple errors are collected."""
        result = self.validator.validate({
            "text": 123,  # Wrong type
            "modality": "invalid",  # Invalid value
            "metadata": "not a dict",  # Wrong type
        })
        self.assertFalse(result.valid)
        self.assertGreaterEqual(len(result.errors), 2)

    def test_error_fields_unique(self):
        """Test that errors reference different fields."""
        result = self.validator.validate({
            "text": 123,
            "modality": 456,
        })
        self.assertFalse(result.valid)
        fields = [e.field for e in result.errors]
        self.assertIn("text", fields)
        self.assertIn("modality", fields)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main()
