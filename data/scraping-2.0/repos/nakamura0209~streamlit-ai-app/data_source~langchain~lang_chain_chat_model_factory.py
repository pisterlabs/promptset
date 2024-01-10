from typing import Any, Dict, Union
from langchain.chat_models import AzureChatOpenAI

from data_source.openai_data_source import MODELS


class ModelParameters:
    def __init__(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        deployment_name: str,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.deployment_name = deployment_name
        pass


class LangchainChatModelFactory:
    @staticmethod
    def create_instance(temperature: float, model: Union[str, Any]) -> AzureChatOpenAI:
        config_key: str = "config"
        """
        NOTE:
        mypyで指摘が入っているが、誤検知と思われる
        継承元のChatOpenAIクラスにはプロパティとして指摘事項の要素を受け取る記載がされている
        """
        return AzureChatOpenAI(
            openai_api_base=MODELS[model][config_key]["base_url"],  # type: ignore
            openai_api_version=MODELS[model][config_key]["api_version"],  # type: ignore
            deployment_name=MODELS[model][config_key]["deployment_name"],  # type: ignore
            openai_api_key=MODELS[model][config_key]["api_key"],  # type: ignore
            openai_api_type=MODELS[model][config_key]["api_type"],
            model_version=MODELS[model][config_key]["model_version"],
            # tiktoken_model_name=os.environ.get("AZURE_OPENAI_TIKTOKEN_MODEL_NAME", "", ""),
            temperature=temperature,
        )
