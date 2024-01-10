

from typing import Type, TypeVar, Optional, Union, Any, Dict, get_args, get_origin
from types import NoneType
from pydantic import BaseModel, ValidationError, model_validator, field_validator
from modules.core.config import BaseConfig
from modules.generation.backend_patterns import Backend, BackendConfig, BACKEND_CLASSES
from modules.generation.generation_patterns import GenerationConfig

# This is _CRITICAL_ - if these aren't imported here, the classes
# can't be loaded down the line! This script is always imported by then
# in normal usage.
from modules.generation.backends.hftgi import HFTGIBackendConfig
from modules.generation.backends.oobabooga import OobaboogaBackendConfig
from modules.generation.backends.openai import OpenAIBackendConfig


# Initialize console logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMConfig(BaseConfig):
    """
    Configuration class for Language Model (LLM) backend and generation settings 
    and parameters.
    Inherits from BaseConfig and defines specific fields relevant to LLM.
    Do NOT use any attributes whose names start with "model_" or it will cause errors

    Attributes:
        backend_config_type (str): Backend provider for the language model.
        backendConfig (BackendConfig): Model backend config object.
        generation_config (GenerationConfig): Generation config object.
    """

    backend_config_type: str
    backend_config: BackendConfig
    generation_preset: Optional[str] = None
    generation_config: GenerationConfig

    @classmethod
    def get_schema_path(cls, data: Optional[Dict[str, Any]] = None, parent_data: Optional[Dict[str, Any]] = None) -> str:
        """Provides the path to the yaml schema for LLMConfig."""
        return "./config/schemas/generation/llm_config.yml"
