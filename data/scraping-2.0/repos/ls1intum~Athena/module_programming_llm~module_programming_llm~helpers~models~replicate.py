import os
from pydantic import Field, PositiveInt
from enum import Enum

from langchain.llms import Replicate
from langchain.base_language import BaseLanguageModel

from athena.logger import logger
from .model_config import ModelConfig


# Hardcoded list of models
# If necessary, add more models from replicate here, the config below might need adjustments depending on the available
# parameters of the model
#
# To update the version of the models, go to the respective page on replicate.com and copy the (latest) version id 
# from and paste it after the colon in the value of the dictionary. Ever so often a new version is released.
replicate_models = {
    # LLAMA 2 70B Chat
    # https://replicate.com/replicate/llama-2-70b-chat
    "llama-2-70b-chat": "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    # LLaMA 2 13B Chat
    # https://replicate.com/a16z-infra/llama-2-13b-chat
    "llama-2-13b-chat": "a16z-infra/llama-2-13b-chat:2a7f981751ec7fdf87b5b91ad4db53683a98082e9ff7bfd12c8cd5ea85980a52",
    # LLaMA 2 7B Chat
    # https://replicate.com/a16z-infra/llama-2-7b-chat
    "llama-2-7b-chat": "a16z-infra/llama-2-7b-chat:7b0bfc9aff140d5b75bacbed23e91fd3c34b01a1e958d32132de6e0a19796e2c",
    # CodeLLAMA 2 13B 
    # https://replicate.com/replicate/codellama-13b
    "codellama-13b": "replicate/codellama-13b:1c914d844307b0588599b8393480a3ba917b660c7e9dfae681542b5325f228db",
    # CodeLLAMA 2 34B
    # https://replicate.com/replicate/codellama-34b
    "codellama-34b": "replicate/codellama-34b:0666717e5ead8557dff55ee8f11924b5c0309f5f1ca52f64bb8eec405fdb38a7",
}

available_models = {}
if os.environ.get("REPLICATE_API_TOKEN"):  # If Replicate is available
    available_models = {
        name: Replicate(
            model=model,
            model_kwargs={ "temperature": 0.01 }
        )
        for name, model in replicate_models.items()
    }
else:
    logger.warning("REPLICATE_API_TOKEN not found in environment variables. Replicate models are disabled.")

if available_models:
    logger.info("Available replicate models: %s",
                ", ".join(available_models.keys()))

    ReplicateModel = Enum('ReplicateModel', {name: name for name in available_models})  # type: ignore


    default_model_name = "llama-2-13b-chat"
    if "LLM_DEFAULT_MODEL" in os.environ and os.environ["LLM_DEFAULT_MODEL"] in available_models:
        default_model_name = os.environ["LLM_DEFAULT_MODEL"]
    if default_model_name not in available_models:
        default_model_name = list(available_models.keys())[0]

    default_replicate_model = ReplicateModel[default_model_name]


    # Note: Config has been setup with LLaMA 2 chat models in mind, other models may not work as expected
    class ReplicateModelConfig(ModelConfig):
        """Replicate LLM configuration."""

        model_name: ReplicateModel = Field(default=default_replicate_model,  # type: ignore
                                           description="The name of the model to use.")
        max_new_tokens: PositiveInt = Field(1000, description="""\
Maximum number of tokens to generate. A word is generally 2-3 tokens (minimum: 1)\
""")
        min_new_tokens: int = Field(-1, description="""\
Minimum number of tokens to generate. To disable, set to -1. A word is generally 2-3 tokens. (minimum: -1)\
""")
        temperature: float = Field(default=0.01, ge=0.01, le=5, description="""\
Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.\
(minimum: 0.01; maximum: 5)\
""")
        top_p: float = Field(default=1, ge=0, le=1, description="""\
When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens (maximum: 1)\
""")
        top_k: PositiveInt = Field(default=250, description="""\
When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens\
""")
        repetition_penalty: float = Field(default=1, ge=0.01, le=5, description="""\
Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, \
less than 1 encourage it. (minimum: 0.01; maximum: 5)\
""")
        repetition_penalty_sustain: int = Field(default=-1, description="""
Number of most recent tokens to apply repetition penalty to, -1 to apply to whole context (minimum: -1)\
""")
        token_repetition_penalty_decay: PositiveInt = Field(default=128, description="""\
Gradually decrease penalty over this many tokens (minimum: 1)\
""")

        def get_model(self) -> BaseLanguageModel:
            """Get the model from the configuration.

            Returns:
                BaseLanguageModel: The model.
            """
            model = available_models[self.model_name.value]
            kwargs = model._lc_kwargs

            model_kwargs = {}
            for attr, value in self.dict().items():
                # Skip model_name
                if attr == "model_name":
                    continue
                model_kwargs[attr] = value

            # Set model parameters
            kwargs["model_kwargs"] = model_kwargs

            # Initialize a copy of the model using the config
            model = model.__class__(**kwargs)
            return model
        

        class Config:
            title = 'Replicate'
