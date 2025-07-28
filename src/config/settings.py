import yaml
from pathlib import Path

class Config:
    def __init__(self, config_file: str = None):
        base_path = Path(__file__).parent  # directory of settings.py
        config_path = base_path / "prompts.yaml" if config_file is None else Path(config_file)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def prompt(self, key: str) -> str:
        return self.config["prompts"][key]
