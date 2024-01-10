import os
import logging
import sys

from dependency_injector import containers, providers
from dotenv import load_dotenv, find_dotenv

from .broker import SemanticBroker
from .jobs import JobManager, JobScheduler
from .activities import (
    ReadActivity,
    WriteActivity,
    SummarizeActivity,
    GenerateActivity,
    ExtractActivity,
    StoreActivity,
    RetrieveActivity,
    FunctionActivity,
    ReturnActivity
)
from .activities.readers import PdfReader, FileReader, ImageReader
from .providers import OpenAIProvider, AzureProvider, AnthropicProvider, LlavaProvider, GeminiProvider
from .http_client import AsyncHttpClient
from .providers import ProviderManager
from .tools.web import WebTool
from .tools.rest import RestTool
from .tools import ToolManager
from .memory import MemoryManager
from .memory.chromadb import ChromaDbRepository


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    logging = providers.Resource(
        logging.basicConfig,
        stream=sys.stdout,
        level=config.log.level,
        format=config.log.format,
    )

    http_client = providers.Singleton(
        AsyncHttpClient
    )

    openai_provider = providers.Singleton(
        OpenAIProvider,
        config=config.providers.openai,
        http_client=http_client
    )

    azure_provider = providers.Singleton(
        AzureProvider,
        config=config.providers.azure,
        http_client=http_client
    )

    anthropic_provider = providers.Singleton(
        AnthropicProvider,
        config=config.providers.anthropic,
        http_client=http_client
    )

    llava_provider = providers.Singleton(
        LlavaProvider,
        config=config.providers.llava,
        http_client=http_client
    )

    gemini_provider = providers.Singleton(
        GeminiProvider,
        config=config.providers.gemini,
        http_client=http_client
    )

    provider_manager = providers.Singleton(
        ProviderManager,
        openai_provider=openai_provider,
        azure_provider=azure_provider,
        anthropic_provider=anthropic_provider,
        llava_provider=llava_provider,
        gemini_provider=gemini_provider
    )

    web_tool = providers.Singleton(
        WebTool,
        config=config.tools.web,
        http_client=http_client
    )

    rest_tool = providers.Singleton(
        RestTool,
        http_client=http_client
    )

    tool_manager = providers.Singleton(
        ToolManager,
        web_tool=web_tool,
        rest_tool=rest_tool
    )

    pdf_reader = providers.Singleton(PdfReader)
    file_reader = providers.Singleton(FileReader)
    image_reader = providers.Singleton(ImageReader)

    read_activity = providers.Singleton(
        ReadActivity,
        file_reader=file_reader,
        pdf_reader=pdf_reader,
        image_reader=image_reader)

    chromadb_repository = providers.Singleton(
        ChromaDbRepository,
        config=config.memory.chromadb
    )

    memory_manager = providers.Singleton(
        MemoryManager,
        chromadb_repository=chromadb_repository
    )

    store_activity = providers.Singleton(
        StoreActivity,
        memory_manager=memory_manager
    )

    retrieve_activity = providers.Singleton(
        RetrieveActivity,
        memory_manager=memory_manager
    )

    write_activity = providers.Singleton(WriteActivity)

    summarize_activity = providers.Singleton(
        SummarizeActivity,
        provider_manager=provider_manager
    )

    generate_activity = providers.Singleton(
        GenerateActivity,
        provider_manager=provider_manager,
        tool_manager=tool_manager
    )

    extract_activity = providers.Singleton(
        ExtractActivity,
        provider_manager=provider_manager
    )

    function_activity = providers.Singleton(FunctionActivity)

    return_activity = providers.Singleton(ReturnActivity)

    job_manager = providers.Singleton(JobManager)

    job_scheduler = providers.Singleton(
        JobScheduler,
        config=config.scheduler,
        job_manager=job_manager,
        read_activity=read_activity,
        write_activity=write_activity,
        summarize_activity=summarize_activity,
        generate_activity=generate_activity,
        extract_activity=extract_activity,
        store_activity=store_activity,
        retrieve_activity=retrieve_activity,
        function_activity=function_activity,
        return_activity=return_activity
    )

    broker = providers.Singleton(
        SemanticBroker,
        job_manager=job_manager,
        job_scheduler=job_scheduler
    )


def get_broker() -> SemanticBroker:
    root = os.path.dirname(find_dotenv())
    load_dotenv()

    container = Container()
    container.config.from_yaml(os.path.join(root, "config.yml"))
    container.init_resources()
    container.wire(modules=[__name__])

    return container.broker()


