"""
Validator - Validates input against schema.

This module performs validation only. It does NOT:
- Sanitize input (that is handled separately)
- Make policy decisions
- Log anything
- Have side effects

Design principles:
- Deterministic
- Stateless
- Returns structured results (never raises uncaught exceptions)
"""

import json
from typing import Any, Dict, List, Optional, Union

from .schema import (
    FieldError,
    FieldSchema,
    RequestSchema,
    ValidationError,
    ValidationErrorType,
    ValidationResult,
    MAX_METADATA_SIZE,
    ALLOWED_MODALITIES,
    ALLOWED_TASK_TYPES,
    ALLOWED_QUALITY_LEVELS,
    ALLOWED_SOURCES,
)


class Validator:
    """
    Validates input data against RequestSchema.

    Usage:
        validator = Validator()
        result = validator.validate({"text": "Hello", "modality": "text"})

        if result.valid:
            process(result.data)
        else:
            handle_errors(result.errors)

    The validator is stateless and can be reused for multiple validations.
    """

    def __init__(self, schema: type = RequestSchema):
        """
        Initialize validator with schema.

        Args:
            schema: Schema class to validate against (default: RequestSchema)
        """
        self.schema = schema

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate input data against schema.

        Args:
            data: Input data to validate (should be dict or string)

        Returns:
            ValidationResult with valid flag, errors, and validated data
        """
        errors: List[FieldError] = []

        # Handle None input
        if data is None:
            errors.append(FieldError(
                field="_input",
                error_type=ValidationErrorType.NULL_VALUE,
                message="Input cannot be null",
                value=None,
            ))
            return ValidationResult(valid=False, errors=errors)

        # Handle string input (convert to dict with text field)
        if isinstance(data, str):
            data = {"text": data}

        # Validate top-level type
        if not isinstance(data, dict):
            errors.append(FieldError(
                field="_input",
                error_type=ValidationErrorType.NOT_A_DICT,
                message=f"Input must be a dictionary, got {type(data).__name__}",
                value=type(data).__name__,
            ))
            return ValidationResult(valid=False, errors=errors)

        # Validate each field
        validated_data: Dict[str, Any] = {}

        for field_name, field_schema in self.schema.FIELDS.items():
            value = data.get(field_name)
            field_errors = self._validate_field(field_name, value, field_schema)

            if field_errors:
                errors.extend(field_errors)
            else:
                # Use validated value or default
                if value is not None:
                    validated_data[field_name] = value
                elif field_schema.default is not None:
                    validated_data[field_name] = field_schema.default

        # Check for unknown fields (warning only, not error)
        known_fields = set(self.schema.FIELDS.keys())
        for key in data.keys():
            if key not in known_fields:
                # Unknown fields are ignored but could be flagged
                pass

        if errors:
            return ValidationResult(valid=False, errors=errors)

        return ValidationResult(valid=True, errors=[], data=validated_data)

    def _validate_field(
        self,
        field_name: str,
        value: Any,
        schema: FieldSchema,
    ) -> List[FieldError]:
        """
        Validate a single field against its schema.

        Returns list of errors (empty if valid).
        """
        errors: List[FieldError] = []

        # Check required field
        if schema.required and value is None:
            errors.append(FieldError(
                field=field_name,
                error_type=ValidationErrorType.MISSING_FIELD,
                message=f"Field '{field_name}' is required",
            ))
            return errors

        # If value is None and field is not required, use default or skip
        if value is None:
            if not schema.allow_null and schema.default is None:
                errors.append(FieldError(
                    field=field_name,
                    error_type=ValidationErrorType.NULL_VALUE,
                    message=f"Field '{field_name}' cannot be null",
                    value=None,
                ))
            return errors

        # Type validation
        type_error = self._validate_type(field_name, value, schema)
        if type_error:
            errors.append(type_error)
            return errors  # Stop validation if type is wrong

        # String-specific validations
        if schema.field_type == str and isinstance(value, str):
            string_errors = self._validate_string(field_name, value, schema)
            errors.extend(string_errors)

        # Dict-specific validations (metadata)
        if schema.field_type == dict and isinstance(value, dict):
            dict_errors = self._validate_dict(field_name, value, schema)
            errors.extend(dict_errors)

        return errors

    def _validate_type(
        self,
        field_name: str,
        value: Any,
        schema: FieldSchema,
    ) -> Optional[FieldError]:
        """
        Validate field type.

        Returns FieldError if type is invalid, None otherwise.
        """
        expected_type = schema.field_type

        # String validation
        if expected_type == str:
            if not isinstance(value, str):
                return FieldError(
                    field=field_name,
                    error_type=ValidationErrorType.NOT_A_STRING,
                    message=f"Field '{field_name}' must be a string, got {type(value).__name__}",
                    value=type(value).__name__,
                    constraint="string",
                )

        # Dict validation
        elif expected_type == dict:
            if not isinstance(value, dict):
                return FieldError(
                    field=field_name,
                    error_type=ValidationErrorType.NOT_A_DICT,
                    message=f"Field '{field_name}' must be a dictionary, got {type(value).__name__}",
                    value=type(value).__name__,
                    constraint="dict",
                )

        # List validation
        elif expected_type == list:
            if not isinstance(value, list):
                return FieldError(
                    field=field_name,
                    error_type=ValidationErrorType.NOT_A_LIST,
                    message=f"Field '{field_name}' must be a list, got {type(value).__name__}",
                    value=type(value).__name__,
                    constraint="list",
                )

        # Number validation (int or float)
        elif expected_type in (int, float):
            if not isinstance(value, (int, float)):
                return FieldError(
                    field=field_name,
                    error_type=ValidationErrorType.NOT_A_NUMBER,
                    message=f"Field '{field_name}' must be a number, got {type(value).__name__}",
                    value=type(value).__name__,
                    constraint="number",
                )

        # Boolean validation
        elif expected_type == bool:
            if not isinstance(value, bool):
                return FieldError(
                    field=field_name,
                    error_type=ValidationErrorType.NOT_A_BOOLEAN,
                    message=f"Field '{field_name}' must be a boolean, got {type(value).__name__}",
                    value=type(value).__name__,
                    constraint="boolean",
                )

        return None

    def _validate_string(
        self,
        field_name: str,
        value: str,
        schema: FieldSchema,
    ) -> List[FieldError]:
        """
        Validate string-specific constraints.

        Returns list of errors (empty if valid).
        """
        errors: List[FieldError] = []

        # Check for null bytes (security issue)
        if "\x00" in value:
            errors.append(FieldError(
                field=field_name,
                error_type=ValidationErrorType.CONTAINS_NULL_BYTES,
                message=f"Field '{field_name}' contains null bytes",
                value="<contains null bytes>",
            ))
            # Continue validation to find all errors

        # Check empty string
        if not schema.allow_empty and value == "":
            errors.append(FieldError(
                field=field_name,
                error_type=ValidationErrorType.EMPTY_VALUE,
                message=f"Field '{field_name}' cannot be empty",
                value="",
            ))

        # Check max length
        if schema.max_length is not None and len(value) > schema.max_length:
            errors.append(FieldError(
                field=field_name,
                error_type=ValidationErrorType.TOO_LONG,
                message=f"Field '{field_name}' exceeds max length of {schema.max_length}",
                value=f"length={len(value)}",
                constraint=schema.max_length,
            ))

        # Check min length
        if schema.min_length is not None and len(value) < schema.min_length:
            errors.append(FieldError(
                field=field_name,
                error_type=ValidationErrorType.TOO_SHORT,
                message=f"Field '{field_name}' is shorter than min length of {schema.min_length}",
                value=f"length={len(value)}",
                constraint=schema.min_length,
            ))

        # Check allowed values
        if schema.allowed_values is not None and value not in schema.allowed_values:
            errors.append(FieldError(
                field=field_name,
                error_type=ValidationErrorType.NOT_IN_ALLOWED_VALUES,
                message=f"Field '{field_name}' has invalid value '{value}'",
                value=value,
                constraint=list(schema.allowed_values),
            ))

        return errors

    def _validate_dict(
        self,
        field_name: str,
        value: Dict[str, Any],
        schema: FieldSchema,
    ) -> List[FieldError]:
        """
        Validate dictionary-specific constraints.

        Used primarily for metadata field validation.
        """
        errors: List[FieldError] = []

        # Check if dict is JSON serializable and within size limit
        try:
            serialized = json.dumps(value)
            if len(serialized) > MAX_METADATA_SIZE:
                errors.append(FieldError(
                    field=field_name,
                    error_type=ValidationErrorType.METADATA_TOO_LARGE,
                    message=f"Field '{field_name}' serialized size exceeds {MAX_METADATA_SIZE} characters",
                    value=f"size={len(serialized)}",
                    constraint=MAX_METADATA_SIZE,
                ))
        except (TypeError, ValueError) as e:
            errors.append(FieldError(
                field=field_name,
                error_type=ValidationErrorType.METADATA_NOT_SERIALIZABLE,
                message=f"Field '{field_name}' is not JSON serializable: {e}",
                value=str(type(value)),
            ))

        return errors

    def validate_text(self, text: Any) -> ValidationResult:
        """
        Convenience method to validate just a text string.

        Args:
            text: Text to validate

        Returns:
            ValidationResult
        """
        return self.validate({"text": text})

    def validate_or_raise(self, data: Any) -> Dict[str, Any]:
        """
        Validate data and raise ValidationError if invalid.

        Args:
            data: Input data to validate

        Returns:
            Validated data dict

        Raises:
            ValidationError: If validation fails
        """
        result = self.validate(data)
        if not result.valid:
            raise ValidationError(result.errors)
        return result.data
