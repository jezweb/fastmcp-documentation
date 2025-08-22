"""
Data Processing Tools
=====================
Tools for data manipulation and processing.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import format_success, format_error, validate_input

logger = logging.getLogger(__name__)


async def process_data(
    data: List[Dict[str, Any]],
    operation: str = "transform",
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process data with specified operation.
    
    Args:
        data: List of data items to process
        operation: Type of operation (transform, filter, aggregate)
        options: Additional processing options
    
    Returns:
        Processed data with operation results
    """
    try:
        # Validate input
        if not data:
            return format_error("No data provided")
        
        if operation not in ["transform", "filter", "aggregate"]:
            return format_error(f"Invalid operation: {operation}")
        
        results = []
        
        if operation == "transform":
            # Apply transformation
            for item in data:
                transformed = {
                    **item,
                    "processed_at": datetime.now().isoformat(),
                    "processing_type": "transform"
                }
                if options and "fields" in options:
                    # Keep only specified fields
                    transformed = {k: transformed.get(k) for k in options["fields"]}
                results.append(transformed)
        
        elif operation == "filter":
            # Apply filtering
            filter_key = options.get("key") if options else None
            filter_value = options.get("value") if options else None
            
            if filter_key:
                results = [
                    item for item in data
                    if item.get(filter_key) == filter_value
                ]
            else:
                results = data
        
        elif operation == "aggregate":
            # Perform aggregation
            if options and "group_by" in options:
                groups = {}
                for item in data:
                    key = item.get(options["group_by"])
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(item)
                
                results = {
                    "groups": groups,
                    "count": len(groups),
                    "total_items": len(data)
                }
            else:
                results = {
                    "count": len(data),
                    "items": data
                }
        
        return format_success({
            "operation": operation,
            "input_count": len(data),
            "output_count": len(results) if isinstance(results, list) else 1,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return format_error(str(e))


async def transform_data(
    data: Dict[str, Any],
    mapping: Dict[str, str],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Transform data structure using field mapping.
    
    Args:
        data: Input data to transform
        mapping: Field mapping (old_key: new_key)
        defaults: Default values for missing fields
    
    Returns:
        Transformed data structure
    """
    try:
        transformed = {}
        
        # Apply mapping
        for old_key, new_key in mapping.items():
            if old_key in data:
                transformed[new_key] = data[old_key]
            elif defaults and new_key in defaults:
                transformed[new_key] = defaults[new_key]
        
        # Add metadata
        transformed["_metadata"] = {
            "transformed_at": datetime.now().isoformat(),
            "original_keys": list(data.keys()),
            "mapped_keys": list(transformed.keys())
        }
        
        return format_success(transformed)
        
    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        return format_error(str(e))


async def validate_data(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validate data against schema.
    
    Args:
        data: Data to validate
        schema: Validation schema
        strict: If True, fail on extra fields
    
    Returns:
        Validation results
    """
    try:
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    actual_type = type(value).__name__
                    if expected_type == "string" and actual_type != "str":
                        errors.append(f"Field '{field}' should be string, got {actual_type}")
                    elif expected_type == "number" and actual_type not in ["int", "float"]:
                        errors.append(f"Field '{field}' should be number, got {actual_type}")
                    elif expected_type == "boolean" and actual_type != "bool":
                        errors.append(f"Field '{field}' should be boolean, got {actual_type}")
                    elif expected_type == "array" and actual_type != "list":
                        errors.append(f"Field '{field}' should be array, got {actual_type}")
                    elif expected_type == "object" and actual_type != "dict":
                        errors.append(f"Field '{field}' should be object, got {actual_type}")
            elif strict:
                warnings.append(f"Unexpected field: {field}")
        
        # Check constraints
        for field, constraints in properties.items():
            if field in data:
                value = data[field]
                
                # Min/max for numbers
                if "minimum" in constraints and value < constraints["minimum"]:
                    errors.append(f"Field '{field}' below minimum: {constraints['minimum']}")
                if "maximum" in constraints and value > constraints["maximum"]:
                    errors.append(f"Field '{field}' above maximum: {constraints['maximum']}")
                
                # Length for strings
                if isinstance(value, str):
                    if "minLength" in constraints and len(value) < constraints["minLength"]:
                        errors.append(f"Field '{field}' too short: min {constraints['minLength']}")
                    if "maxLength" in constraints and len(value) > constraints["maxLength"]:
                        errors.append(f"Field '{field}' too long: max {constraints['maxLength']}")
                
                # Pattern matching
                if "pattern" in constraints and isinstance(value, str):
                    import re
                    if not re.match(constraints["pattern"], value):
                        errors.append(f"Field '{field}' doesn't match pattern: {constraints['pattern']}")
        
        is_valid = len(errors) == 0
        
        return format_success({
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "data": data if is_valid else None
        })
        
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return format_error(str(e))


async def export_data(
    data: List[Dict[str, Any]],
    format: str = "json",
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Export data in specified format.
    
    Args:
        data: Data to export
        format: Export format (json, csv, xml)
        options: Format-specific options
    
    Returns:
        Exported data in specified format
    """
    try:
        if format == "json":
            # JSON export
            indent = options.get("indent", 2) if options else 2
            exported = json.dumps(data, indent=indent, default=str)
            
        elif format == "csv":
            # CSV export
            import csv
            import io
            
            if not data:
                return format_error("No data to export")
            
            # Get all unique keys
            keys = set()
            for item in data:
                keys.update(item.keys())
            keys = sorted(keys)
            
            # Create CSV
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
            
            exported = output.getvalue()
            
        elif format == "xml":
            # Simple XML export
            lines = ['<?xml version="1.0" encoding="UTF-8"?>']
            lines.append('<data>')
            
            for item in data:
                lines.append('  <item>')
                for key, value in item.items():
                    # Escape XML special characters
                    value_str = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    lines.append(f'    <{key}>{value_str}</{key}>')
                lines.append('  </item>')
            
            lines.append('</data>')
            exported = '\n'.join(lines)
            
        else:
            return format_error(f"Unsupported format: {format}")
        
        return format_success({
            "format": format,
            "size": len(exported),
            "records": len(data),
            "content": exported
        })
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return format_error(str(e))