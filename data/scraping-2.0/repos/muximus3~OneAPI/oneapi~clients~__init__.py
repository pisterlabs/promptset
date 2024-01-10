from oneapi.clients.abc_client import AbstractClient, AbstractConfig
from oneapi.clients.anthropic_client import AnthropicClient, AnthropicConfig, AnthropicDecodingArguments
from oneapi.clients.hf_client import HuggingfaceClient, HuggingFaceConfig, HuggingFaceDecodingArguments
from oneapi.clients.vllm_client import VLLMClient, VLLMConfig, VLLMDecodingArguments
from oneapi.clients.openai_client import OpenAIClient, OpenAIConfig, OpenAIDecodingArguments

clients_register = {
    "claude": AnthropicClient,
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "open_ai": OpenAIClient,
    "azure": OpenAIClient,
    "huggingface": HuggingfaceClient,
    "vllm": VLLMClient
    }