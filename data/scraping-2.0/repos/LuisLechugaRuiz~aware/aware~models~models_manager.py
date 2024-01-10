import logging

from aware.config import get_modules
from aware.models.model import Model

from aware.models.open.os_model import OSModel
from aware.models.private.openai import OpenAIModel

LOG = logging.getLogger(__name__)


class ModelsManager:
    """Manager to ensure that the models are able to run properly on GPU"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelsManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):  # Avoid re-initialization
            self.models = {}
            self._initialized = True

    def create_model(self, module_name: str) -> "Model":
        """Create a model for a given module depending on the type"""

        modules_config = get_modules()
        model_config = modules_config[module_name]["model"]
        model_type = model_config["type"]
        model_name = model_config["name"]

        if model_type == "openai":
            model = OpenAIModel(model_name=model_name)
        elif model_type == "open_source":
            # 1. Check GPU, verify if we should unload any model - Iterate over self.models and unload if necessary.

            # !! TEMPORALLY JUST UNLOADING ALL OPEN SOURCE MODELS !!!
            os_models = []
            for name, model in self.models.items():
                # Check if model is a OSModel
                if type(model) is OSModel:
                    os_models.append(name)
            # self.unload_models(os_models)

            # 2. Load model
            model = OSModel(
                module_name=module_name,
                model_name=model_name,
                model_revision=model_config.get("revision", None),
            )
        else:
            raise ValueError(f"Model type {model_type} not recognized.")
        self.models[module_name] = model
        return model

    # TODO: Call this function before running the system and save the specific config for the user.
    def get_model_config(self):
        """Fetch user GPU and get the best config for them"""  # Initially from specific config files, in the future hopefully we can fetch them automatically from HuggingFace.
        pass

    # TODO: Call this function before running the system and save the specific config for the user.
    def verify_compatibility(self):
        """Set config which determines which models can be running concurrently."""
        pass
