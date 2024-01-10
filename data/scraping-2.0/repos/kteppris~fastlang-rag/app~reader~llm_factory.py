from __future__ import annotations
from omegaconf import OmegaConf
import os
from langchain.llms import CTransformers, HuggingFaceEndpoint, HuggingFacePipeline, HuggingFacePipeline, OpenAI

class LLMFactory:
    """
    Factory for producing LLM (Language Learning Model) instances based on configuration.
    """
    @staticmethod
    def get_llm(config) -> HuggingFaceEndpoint | CTransformers | OpenAI:
        """
        Factory method to get the initialized LLM based on the provided configuration.

        Parameters
        ----------
        config : OmegaConf
            Configuration specifying the type of LLM and its initialization parameters.

        Returns
        -------
        Any
            An initialized LLM. The exact type and behavior depends on the configuration.

        Raises
        ------
        ValueError
            If the LLM type specified in the configuration is not supported.
        """
        if "huggingface" or "ctransformers" in config.reader.type.lower():
            return LLMFactory._get_huggingface_llm(config)
        elif config.reader.type == "OpenAIReader":
            return LLMFactory._get_openai_reader(config)
        else:
            raise ValueError(f"Unknown reader type: {config.reader.type}")

    @staticmethod
    def _get_huggingface_llm(config) -> HuggingFaceEndpoint | CTransformers:
        """
        (Private) Initializes and returns an LLM based on Huggingface configurations.

        Parameters
        ----------
        config : OmegaConf
            Configuration for Huggingface LLM initialization.

        Returns
        -------
        Any
            An initialized LLM.
        """
        model_id = f"{config.reader.user}/{config.reader.name}"
        if config.reader.type == "HuggingfaceEndpoint":
            return HuggingFaceEndpoint(
                endpoint_url= config.reader.endpoint + config.reader.name,
                task="text-generation",
                model_kwargs=OmegaConf.to_container(config.reader.args.model_kwargs)
            )
        elif config.reader.type == "CTransformers":
            return CTransformers(
                model=model_id,
                **config.reader.args
            )
        else:
            return HuggingFacePipeline.from_model_id(
                model_id=model_id,
                task="text-generation",
                **config.reader.args
            )

    @staticmethod
    def _get_openai_reader(config) -> OpenAI:
        """
        (Private) Initializes and returns an LLM based on OpenAI configurations.

        Parameters
        ----------
        config : OmegaConf
            Configuration for OpenAI LLM initialization.

        Returns
        -------
        Any
            An initialized LLM.
        """
        
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        return OpenAI(
            model_name=config.reader.name
        )
