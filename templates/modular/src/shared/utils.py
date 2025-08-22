"""
Utility Functions
=================
Common utility functions used across the server.
"""

import re
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def format_success(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
    """
    Format a successful response.
    
    Args:
        data: Response data
        message: Optional success message
    
    Returns:
        Formatted success response
    """
    response = {
        "success": True,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    if message:
        response["message"] = message
    
    return response


def format_error(
    error: Union[str, Exception],
    code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format an error response.
    
    Args:
        error: Error message or exception
        code: Optional error code
        details: Optional error details
    
    Returns:
        Formatted error response
    """
    response = {
        "success": False,
        "error": str(error),
        "timestamp": datetime.now().isoformat()
    }
    
    if code:
        response["code"] = code
    
    if details:
        response["details"] = details
    
    # Log the error
    logger.error(f"Error response: {response}")
    
    return response


def validate_input(
    data: Any,
    schema: Dict[str, Any],
    strict: bool = False
) -> tuple[bool, Optional[str]]:
    """
    Validate input data against a schema.
    
    Args:
        data: Data to validate
        schema: Validation schema
        strict: If True, reject extra fields
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check field types
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                field_schema = properties[field]
                expected_type = field_schema.get("type")
                
                if expected_type:
                    # Type validation
                    if expected_type == "string" and not isinstance(value, str):
                        return False, f"Field '{field}' must be a string"
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        return False, f"Field '{field}' must be a number"
                    elif expected_type == "integer" and not isinstance(value, int):
                        return False, f"Field '{field}' must be an integer"
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        return False, f"Field '{field}' must be a boolean"
                    elif expected_type == "array" and not isinstance(value, list):
                        return False, f"Field '{field}' must be an array"
                    elif expected_type == "object" and not isinstance(value, dict):
                        return False, f"Field '{field}' must be an object"
                
                # Additional constraints
                if isinstance(value, str):
                    # String constraints
                    if "minLength" in field_schema and len(value) < field_schema["minLength"]:
                        return False, f"Field '{field}' is too short (min: {field_schema['minLength']})"
                    if "maxLength" in field_schema and len(value) > field_schema["maxLength"]:
                        return False, f"Field '{field}' is too long (max: {field_schema['maxLength']})"
                    if "pattern" in field_schema:
                        if not re.match(field_schema["pattern"], value):
                            return False, f"Field '{field}' doesn't match required pattern"
                    if "enum" in field_schema and value not in field_schema["enum"]:
                        return False, f"Field '{field}' must be one of: {field_schema['enum']}"
                
                elif isinstance(value, (int, float)):
                    # Number constraints
                    if "minimum" in field_schema and value < field_schema["minimum"]:
                        return False, f"Field '{field}' is below minimum: {field_schema['minimum']}"
                    if "maximum" in field_schema and value > field_schema["maximum"]:
                        return False, f"Field '{field}' is above maximum: {field_schema['maximum']}"
                
                elif isinstance(value, list):
                    # Array constraints
                    if "minItems" in field_schema and len(value) < field_schema["minItems"]:
                        return False, f"Field '{field}' has too few items (min: {field_schema['minItems']})"
                    if "maxItems" in field_schema and len(value) > field_schema["maxItems"]:
                        return False, f"Field '{field}' has too many items (max: {field_schema['maxItems']})"
            
            elif strict:
                return False, f"Unexpected field: {field}"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, str(e)


def validate_path(path: str, must_exist: bool = False) -> bool:
    """
    Validate a file system path.
    
    Args:
        path: Path to validate
        must_exist: If True, path must exist
    
    Returns:
        True if path is valid
    """
    try:
        # Check for path traversal attempts
        if ".." in path or path.startswith("/"):
            logger.warning(f"Path traversal attempt detected: {path}")
            return False
        
        # Convert to Path object
        p = Path(path)
        
        # Check if path exists if required
        if must_exist and not p.exists():
            return False
        
        # Resolve path (follows symlinks)
        resolved = p.resolve()
        
        # Ensure resolved path is within allowed directory
        # In production, you'd check against a configured base directory
        # For now, just ensure it's not accessing system directories
        restricted_dirs = ["/etc", "/sys", "/proc", "/dev", "/root"]
        for restricted in restricted_dirs:
            if str(resolved).startswith(restricted):
                logger.warning(f"Access to restricted directory attempted: {resolved}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Path validation error: {e}")
        return False


def sanitize_string(
    text: str,
    max_length: Optional[int] = None,
    allowed_chars: Optional[str] = None
) -> str:
    """
    Sanitize a string for safe use.
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        allowed_chars: Regex pattern of allowed characters
    
    Returns:
        Sanitized string
    """
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    
    # Apply character filter if specified
    if allowed_chars:
        text = re.sub(f"[^{allowed_chars}]", "", text)
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def parse_duration(duration_str: str) -> int:
    """
    Parse duration string to seconds.
    
    Args:
        duration_str: Duration string (e.g., "5m", "1h", "30s")
    
    Returns:
        Duration in seconds
    """
    match = re.match(r'^(\d+)([smhd])$', duration_str.lower())
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    multipliers = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400
    }
    
    return value * multipliers[unit]


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        bytes_value: Number of bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def truncate_text(
    text: str,
    max_length: int = 100,
    suffix: str = "..."
) -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append when truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """
    Merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        deep: If True, perform deep merge
    
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    if not deep:
        result.update(dict2)
        return result
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value
    
    return result