from typing import Dict, Any, List, Optional, Callable

def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate a configuration dictionary against a schema."""
    errors = []
    
    for key, spec in schema.items():
        # Check required fields
        if spec.get("required", False) and key not in config:
            errors.append(f"Missing required field: {key}")
            continue
            
        # Skip validation if the field isn't present
        if key not in config:
            continue
            
        value = config[key]
        
        # Check type
        if "type" in spec:
            expected_type = spec["type"]
            if not isinstance(value, expected_type):
                errors.append(f"Field {key} should be of type {expected_type.__name__}, got {type(value).__name__}")
        
        # Check enum values
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"Field {key} must be one of {spec['enum']}, got {value}")
        
        # Check custom validation
        if "validate" in spec and callable(spec["validate"]):
            validation_result = spec["validate"](value)
            if validation_result is not True:
                errors.append(f"Field {key} validation failed: {validation_result}")
    
    return errors

def validate_message_format(message: Dict[str, Any]) -> List[str]:
    """Validate a message follows the correct format."""
    schema = {
        "id": {"required": True, "type": str},
        "sender_id": {"required": True, "type": str},
        "receiver_id": {"required": True, "type": str},
        "content": {"required": True, "type": str},
        "message_type": {"required": True, "type": str},
        "conversation_id": {"required": True, "type": str},
        "timestamp": {"required": True, "type": str}
    }
    
    return validate_config(message, schema)