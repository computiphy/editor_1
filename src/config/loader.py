import yaml
from src.config.schema import ConfigSchema

def load_config(file_path: str) -> ConfigSchema:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return ConfigSchema(**data)
