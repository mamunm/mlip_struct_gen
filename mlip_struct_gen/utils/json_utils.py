"""Utilities for saving parameters to JSON files."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def save_parameters_to_json(parameters: Any, filepath: str = "input_params.json") -> None:
    """
    Save dataclass parameters to a JSON file.

    Args:
        parameters: Dataclass instance containing parameters
        filepath: Path to save the JSON file (default: input_params.json)
    """
    if not is_dataclass(parameters):
        raise TypeError("Parameters must be a dataclass instance")

    # Convert dataclass to dictionary
    params_dict = asdict(parameters)

    # Remove logger if present (not serializable)
    if "logger" in params_dict:
        params_dict["logger"] = None

    # Convert Path objects to strings
    params_dict = _convert_paths_to_strings(params_dict)

    # Save to JSON file
    output_path = Path(filepath)
    with open(output_path, "w") as f:
        json.dump(params_dict, f, indent=2, default=str)

    print(f"Input parameters saved to: {output_path}")


def _convert_paths_to_strings(obj: Any) -> Any:
    """
    Recursively convert Path objects to strings in nested structures.

    Args:
        obj: Object to convert

    Returns:
        Object with Path instances converted to strings
    """
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _convert_paths_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_paths_to_strings(item) for item in obj)
    else:
        return obj


def load_parameters_from_json(filepath: str, parameter_class: type) -> Any:
    """
    Load parameters from a JSON file and create a dataclass instance.

    Args:
        filepath: Path to the JSON file
        parameter_class: The dataclass type to instantiate

    Returns:
        Instance of parameter_class with loaded values
    """
    with open(filepath) as f:
        params_dict = json.load(f)

    # Remove None values for logger
    if "logger" in params_dict:
        params_dict.pop("logger", None)

    return parameter_class(**params_dict)
