from unittest.mock import patch, Mock

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from prompts.sql_runner import SQLRunner, Query
from utils.database import create_db_session
from utils.config_loaders import (
    get_config,
    get_secrets
)

config = get_config()
secrets = get_secrets(config)


class TestQuery:

    def test_query_fields(self):
        table = Query(query="SELECT * FROM public.example")
        assert table.query == 'SELECT * FROM public.example'
        assert 'query' in Query.schema()['properties']


class TestSQLRunnerGeneral:
    def test_sql_runner_fields(self):
        assert 'input' in SQLRunner.schema()['properties']
        assert 'table_info' in SQLRunner.schema()['properties']
        assert 'dialect' in SQLRunner.schema()['properties']
        assert 'top_k' in SQLRunner.schema()['properties']
        assert 'model_name' in SQLRunner.schema()['properties']
        assert 'temperature' in SQLRunner.schema()['properties']


class TestSpanishSQLRunner:
    def test_sql_runner(self):
        engine = create_db_session(secrets['database']['url'])

        table_metadata = {
            'public.example': {
                'table': 'public.example',
                'columns': [
                    {
                        'column_name': 'primary_key',
                        'type': 'BIGINT',
                        'comment': None,
                        'is_foreign_key': False,
                        'is_primary_key': True,
                        'foreign_key_tables': None
                    }]
            }
        }
        runner = SQLRunner(
            input='I want the count of records from the example table in the public schema.',
            table_info=table_metadata,
            dialect=engine.dialect.name,
            top_k=30,
            model_name='gpt-3.5-turbo',
            temperature=0
        )
        result = runner.get_result()
        assert result['query'] in ('SELECT COUNT(*) FROM public.example;', 'SELECT COUNT(*) FROM public.example')