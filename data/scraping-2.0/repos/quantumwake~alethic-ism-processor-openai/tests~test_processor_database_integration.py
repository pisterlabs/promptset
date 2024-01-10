import os

from core.processor_state import State, StateConfig, StateDataKeyDefinition, StateConfigLM
from db.misc_utils import create_state_id_by_config
from db.processor_state_db import ProcessorStateDatabaseStorage

from processor_question_answer import OpenAIQuestionAnswerProcessor

test_state_database_url = os.environ.get("STATE_DATABASE_URL", "postgresql://postgres:postgres1@localhost:5432/postgres")


def test_processor_database_state_storage_integration():

    # build a mock input state
    test_state = State(
        config=StateConfig(
            name="Processor Question Answer OpenAI (Test)",
            version="Test Version 0.0",
            primary_key=[
                StateDataKeyDefinition(name="query"),
                StateDataKeyDefinition(name="animal")
            ]
        )
    )

    test_queries = [
        {"query": "tell me about cats.", "animal": "cat"},
        {"query": "tell me about pigs.", "animal": "pig"},
        {"query": "what is a cat?", "animal": "cat"},
        {"query": "what is a pig?", "animal": "pig"}
    ]

    # apply the data to the mock input state
    for test_query in test_queries:
        test_state.apply_columns(test_query)
        test_state.apply_row_data(test_query)

    # persist the mocked input state into the database
    test_storage = ProcessorStateDatabaseStorage(database_url=test_state_database_url)
    test_storage.save_state(state=test_state)

    # reload the newly persisted state, for testing purposes
    test_state_id = create_state_id_by_config(config=test_state.config)
    test_state = test_storage.load_state(state_id=test_state_id)

    # process the input state
    test_processor = OpenAIQuestionAnswerProcessor(
        state=State(
            config=StateConfigLM(
                name="Processor Question Answer OpenAI (Test Response State)",
                version="Test Version 0.0",
                model_name="gpt-4-1106-preview",
                provider_name="OpenAI",
                storage_class="database",
                primary_key=[
                    StateDataKeyDefinition(name="query"),
                    StateDataKeyDefinition(name="animal")
                ],
                query_state_inheritance=[
                    StateDataKeyDefinition(name="query", alias="response_query"),
                    StateDataKeyDefinition(name="animal", alias="animal")
                ],
                user_template_path="./test_templates/test_template_P1_user.json",
                system_template_path="./test_templates/test_template_P1_system.json",
            )
        )
    )
    test_processor(input_state=test_state)