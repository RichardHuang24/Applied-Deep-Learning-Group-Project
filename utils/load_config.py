import json
from pathlib import Path


def load_config(config_path):
    """
    Load JSON config from file.

    Args:
        config_path (str or Path): Path to config file

    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config
