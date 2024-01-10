from dependency_injector import containers, providers
from tussle.completions.providers.db_cache_completion_provider_test import DbCacheCompletionProvider
from tussle.completions.providers.memory_cache_completion_provider import MemoryCacheCompletionProvider
from tussle.completions.providers.openai_completion_provider import OpenAICompletionProvider
from tussle.completions.providers.magic_string_test_prompt_completion_provider import MagicStringTestPromptCompletionProvider
from tussle.completions.providers.combined_completion_provider import CombinedCompletionProvider
from tussle.integrations.mongo.mongo_integration import connect_to_database_pymongo
from tussle.general.config.config import load_cloud_configuration, configure_environment_variables
from tussle.integrations.openai.openai_integration import OpenAIIntegration

from tussle.debate.data_repository.mongo.mongo_topic_repository import MongoTopicRepository
from tussle.debate.data_repository.mongo.mongo_answer_repository import MongoAnswerRepository


class Application(containers.DeclarativeContainer):
    environment_variables = providers.ThreadSafeSingleton(
        configure_environment_variables,
    )

    config = providers.ThreadSafeSingleton(
        load_cloud_configuration,
    )

    mongo_db_connection = providers.Resource(
        connect_to_database_pymongo,
        config,
    )

    openai_integration = providers.Resource(
        OpenAIIntegration,
        config,
    )

    memory_cache_completion_provider = providers.ThreadSafeSingleton(
        MemoryCacheCompletionProvider,
    )

    magic_string_test_completion_provider = providers.ThreadSafeSingleton(
        MagicStringTestPromptCompletionProvider,
    )

    db_cache_completion_provider = providers.ThreadSafeSingleton(
        DbCacheCompletionProvider,
        mongo_db_connection,
    )

    openai_completion_provider = providers.ThreadSafeSingleton(
        OpenAICompletionProvider,
        openai_integration,
    )

    # If OpenAI is disabled, then we just wire the main completion provider
    # as the lorem ipsum provider. Otherwise, we wire it as the combined
    # provider
    combined_completion_provider = providers.ThreadSafeSingleton(
        CombinedCompletionProvider,
        memory_cache_completion_provider,
        db_cache_completion_provider,
        magic_string_test_completion_provider,
        openai_completion_provider
    )

    completion_provider = providers.Selector(
        lambda **kwargs: 'combined',
        combined=combined_completion_provider,
    )

    answer_repository = providers.ThreadSafeSingleton(
        MongoAnswerRepository,
        mongo_db_connection,
    )

    topic_repository = providers.ThreadSafeSingleton(
        MongoTopicRepository,
        mongo_db_connection,
    )




def create_and_initialization_application_container():
    from tussle.general.testing.initialization import setup_test_overrides_if_needed

    container = Application()

    # Make sure the container has the correct overrides in place for testing.
    setup_test_overrides_if_needed(container)

    container.init_resources()

    # Here we explicitly wire the packages that matter, so that it doesn't try to search and load everything.
    # This is to improve the server load time.
    container.wire(modules=[
        'tussle.debate.apis.answer_api',
        'tussle.debate.apis.topic_api',
        'tussle.completions.utils.completion_utils',
        'tussle.completions.apis.completion_api',
        'tussle.general.api_server.permissions',
        'tussle.general.components.health_check.db_access_health_check',
        'tussle.general.components.health_check.openai_integration_health_check',
        'tussle.general.db_models.custom_id_field',
    ])

    return container
