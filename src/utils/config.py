"""
Configuration loader — reads pipeline_config.yaml and env vars.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PipelineConfig:
    """Typed wrapper around the YAML configuration."""
    raw: Dict[str, Any] = field(default_factory=dict)

    # Quick accessors
    @property
    def ingestion(self) -> dict:
        return self.raw.get("ingestion", {})

    @property
    def embedding(self) -> dict:
        return self.raw.get("embedding", {})

    @property
    def retrieval(self) -> dict:
        return self.raw.get("retrieval", {})

    @property
    def generation(self) -> dict:
        return self.raw.get("generation", {})

    @property
    def security(self) -> dict:
        return self.raw.get("security", {})

    @property
    def observability(self) -> dict:
        return self.raw.get("observability", {})

    @property
    def deployment(self) -> dict:
        return self.raw.get("deployment", {})

    def get(self, dotpath: str, default=None):
        """Access nested config via dot notation: 'embedding.model.provider'."""
        keys = dotpath.split(".")
        val = self.raw
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val


def _resolve_env_vars(obj):
    """Recursively replace ${ENV_VAR} patterns with environment values."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            return os.environ.get(obj[2:-1], obj)
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.
    
    Resolution order:
      1. Explicit path argument
      2. LLMRAG_CONFIG environment variable
      3. ./configs/pipeline_config.yaml
      4. ./pipeline_config.yaml
    """
    search_paths = [
        config_path,
        os.environ.get("LLMRAG_CONFIG"),
        str(Path.cwd() / "configs" / "pipeline_config.yaml"),
        str(Path.cwd() / "pipeline_config.yaml"),
        str(Path(__file__).parent.parent.parent / "configs" / "pipeline_config.yaml"),
    ]

    for p in search_paths:
        if p and Path(p).exists():
            with open(p, "r") as f:
                raw = yaml.safe_load(f)
            raw = _resolve_env_vars(raw)
            return PipelineConfig(raw=raw)

    # Return defaults if no config file found
    print("⚠  No config file found — using defaults.")
    return PipelineConfig(raw={})
