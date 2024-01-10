from llama_index import ServiceContext
from llama_index.llms import OpenAI


class ServiceContextFactory:
    @staticmethod
    def create_service_context(config):
        llm_service = config.metadata_extractor.llm_service
        llm_kwargs = config.metadata_extractor.llm_kwargs

        if llm_service == "OpenAI":
            llm = OpenAI(**llm_kwargs)
        elif llm_service == "AnotherService":
            # Initialize another service here
            pass
        else:
            raise ValueError(f"Unsupported llm service: {llm_service}")

        return ServiceContext.from_defaults(llm=llm)
