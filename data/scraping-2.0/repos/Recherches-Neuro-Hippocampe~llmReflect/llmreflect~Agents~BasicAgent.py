from llmreflect.Retriever.BasicRetriever import BasicRetriever
from abc import ABC, abstractclassmethod
from llmreflect.LLMCore.LLMCore import LLMCore, OPENAI_MODEL
from llmreflect.LLMCore.LLMCore import OpenAICore, LlamacppCore
from typing import Any


class BasicAgent(ABC):
    PROMPT_NAME = ""  # the name of prompt used

    def __init__(self, llm_core: LLMCore, **kwargs) -> None:
        """
        In this design each agent should have
        a retriever, retriever is for retrieving the final result based
        on the gross output by LLM.
        For example, a database retriever does the following job:
        extract the sql command from the llm output and then
        execute the command in the database.
        """
        object.__setattr__(self, 'llm_core', llm_core)
        object.__setattr__(self, 'local', False)
        if len(self.PROMPT_NAME) == 0:
            raise NotImplementedError(
                "PROMPT_NAME has to be implemented for each agent!")

    def equip_retriever(self, retriever: BasicRetriever):
        """Equip retriever for an agent object

        Args:
            retriever (BasicRetriever): A retriever instance.
        """
        object.__setattr__(self, 'retriever', retriever)

    @classmethod
    @abstractclassmethod
    def predict(self, **kwargs: Any) -> Any:
        """Use LLM to predict, core function of an agent.

        Raises:
            NotImplementedError: Abstract method has to be implemented.

        Returns:
            Any: llm result.
        """
        raise NotImplementedError("Need to be implemented!")

    @classmethod
    @abstractclassmethod
    def from_config(cls, **kwargs) -> Any:
        """Initialize an instance from config

        Raises:
            NotImplementedError: Abstract method has to be implemented.

        Returns:
            Any: an instance of BasicAgent class.
        """
        raise NotImplementedError("Need to be implemented!")


class Agent(BasicAgent):
    def __init__(self, llm_core: LLMCore, **kwargs) -> None:
        """Initialization for Agent class.

        Args:
            llm_core (LLMCore): An instance of LLMCore class
                Used for inferencing.
        """
        super().__init__(llm_core=llm_core, **kwargs)

    @classmethod
    def from_config(
            cls,
            llm_config: dict = {},
            other_config: dict = {}) -> BasicAgent:
        """Initialize an instance of Agent class from config.

        Args:
            llm_config (dict, optional): Configuration for the LLMCore.
                Usually, in llmreflect, the Agent class is always used
                inside a Chain class.
                To know more about this config of your chain,
                You can always use `NameOfYourChain.local_config_dict` or
                `NameOfYourChain.openai_config_dict`
                Also, it is used for either `init_llm_from_llama` or
                `init_llm_from_openai` functions.
                Defaults to {}.
            other_config (dict, optional): Configuration other than llm_config
                Defaults to {}.

        Returns:
            BasicAgent: An instance of the Agent class.
        """
        if "prompt_name" in other_config.keys():
            prompt_name = other_config.get("prompt_name")
            del other_config["prompt_name"]
        else:
            prompt_name = cls.PROMPT_NAME

        if "model_path" in llm_config.keys():
            llm_core = cls.init_llm_from_llama(prompt_name=prompt_name,
                                               **llm_config)
        else:
            llm_core = cls.init_llm_from_openai(prompt_name=prompt_name,
                                                **llm_config)
        return cls(llm_core=llm_core,
                   **other_config)

    @classmethod
    def init_llm_from_llama(cls,
                            model_path: str,
                            prompt_name: str,
                            max_total_tokens: int = 4096,
                            max_output_tokens: int = 512,
                            temperature: float = 0.,
                            verbose: bool = False,
                            n_gpus_layers: int = 8,
                            n_threads: int = 16,
                            n_batch: int = 512) -> BasicAgent:
        """
        Return a llama.cpp llmcore instance.
        Args:
            model_path (str): path to the model file,
                options can be found in LLMCORE.LLMCORE.LOCAL_MODEL
            prompt_name (str): name of the prompt file
            max_total_tokens (int, optional): the entire context length.
                Defaults to 4096.
            max_output_tokens (int, optional): the maximum length
                for the completion. Defaults to 512.
            temperature (float, optional): The temperature to use for
                sampling. The lower the stabler. Defaults to 0.
            verbose (bool, optional): whether to show. Defaults to False.
            n_gpus_layers (int, optional): Number of layers to be loaded
                into gpu memory. Defaults to 8.
            n_threads (int, optional): Number of threads to use.
                Defaults to 16.
            n_batch (int, optional): Maximum number of prompt tokens to batch
                together when calling llama_eval. Defaults to 512.

        Returns:
            BasicAgent: Return a llama.cpp llmcore instance.
        """
        return LlamacppCore(
            model_path=model_path,
            prompt_name=prompt_name,
            max_total_tokens=max_total_tokens,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            verbose=verbose,
            n_gpus_layers=n_gpus_layers,
            n_threads=n_threads,
            n_batch=n_batch
        )

    @classmethod
    def init_llm_from_openai(
            cls,
            open_ai_key: str,
            prompt_name: str,
            max_output_tokens: int = 512,
            temperature: float = 0.0,
            llm_model=OPENAI_MODEL.gpt_3_5_turbo) -> BasicAgent:
        """
        Return an openai llmcore instance.
        Args:
            open_ai_key (str): OpenAI key
            prompt_name (str, optional): name for the prompt. Defaults to ''.
            max_output_tokens (int, optional): maximum number of output tokens.
                Defaults to 512.
            temperature (float, optional): Flexibility of the output.
                Defaults to 0.0.
            llm_model (str, optional): string indicating the mode to use.
                Should be included in class LLM_BACKBONE_MODEL.
                Defaults to LLM_BACKBONE_MODEL.gpt_3_5_turbo.

        Returns:
            BasicAgent: Return an openai llmcore instance.
        """

        return OpenAICore(
            open_ai_key=open_ai_key,
            prompt_name=prompt_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            llm_model=llm_model
        )

    def predict(self, **kwargs: Any) -> str:
        return self.llm_core.predict(**kwargs)
