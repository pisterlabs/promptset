from typing import Mapping

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.llm.chat_model import ChatModel, Model
from aidial_adapter_bedrock.llm.model.ai21 import AI21Adapter
from aidial_adapter_bedrock.llm.model.amazon import AmazonAdapter
from aidial_adapter_bedrock.llm.model.anthropic import AnthropicAdapter
from aidial_adapter_bedrock.llm.model.cohere import CohereAdapter
from aidial_adapter_bedrock.llm.model.meta import MetaAdapter
from aidial_adapter_bedrock.llm.model.stability import StabilityAdapter


async def get_bedrock_adapter(
    model: str, region: str, headers: Mapping[str, str]
) -> ChatModel:
    client = await Bedrock.acreate(region)
    provider = Model.parse(model).provider
    match provider:
        case "anthropic":
            return AnthropicAdapter.create(client, model)
        case "ai21":
            return AI21Adapter.create(client, model)
        case "stability":
            return StabilityAdapter.create(client, model, headers)
        case "amazon":
            return AmazonAdapter.create(client, model)
        case "meta":
            return MetaAdapter.create(client, model)
        case "cohere":
            return CohereAdapter.create(client, model)
        case _:
            raise ValueError(f"Unknown model provider: '{provider}'")
