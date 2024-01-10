from embeddings.spaces_text import SpacesGradioTextEmbeddings
from embeddings.base import EmbeddingsGenerator
from embeddings.spaces_instruct import SpacesGradioInstructTextEmbeddings
from embeddings.openai import OpenAIEmbeddings, OpenAISettings, AzureOpenAISettings
from common.utils import get_env_or_fail


class EmbeddingsGeneratorProvider:
    @staticmethod
    def get_client(name: str) -> EmbeddingsGenerator:
        if name == 'SPACES_TEXT':
            return SpacesGradioTextEmbeddings(
                endpoint_url=get_env_or_fail("SPACES_URL"),
                endpoint_key=get_env_or_fail("SPACES_KEY"),
                batch_size=int(get_env_or_fail("BATCH_SIZE")),
            )
        if name == 'SPACES_INSTRUCT':
            return SpacesGradioInstructTextEmbeddings(
                endpoint_url=get_env_or_fail("SPACES_URL"),
                endpoint_key=get_env_or_fail("SPACES_KEY"),
                batch_size=int(get_env_or_fail("BATCH_SIZE")),
            )
        if name == 'OPENAI':
            type = get_env_or_fail("OPENAI_TYPE")
            if type not in ['openai', 'azure']:
                raise Exception("Invalid OpenAI type")

            return OpenAIEmbeddings(settings=OpenAISettings(
                type= 'openai' if type == 'openai' else 'azure',
                key=get_env_or_fail("OPENAI_API_KEY"),
                model=get_env_or_fail("OPENAI_MODEL"),
                batch_size=int(get_env_or_fail("BATCH_SIZE")),
                azure_settings=None if type == 'openai' else AzureOpenAISettings(
                    api_endpoint=get_env_or_fail("AZURE_OPENAI_ENDPOINT"),
                    api_version=get_env_or_fail("AZURE_OPENAI_VERSION"),
                    useActiveDirectory=get_env_or_fail("AZURE_OPENAI_USE_ACTIVE_DIRECTORY").lower() == 'true',
                )
            ))

        raise Exception("Missing valid embedding generator client name")
