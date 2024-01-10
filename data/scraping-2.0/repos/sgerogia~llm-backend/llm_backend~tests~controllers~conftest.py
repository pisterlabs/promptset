import logging
import os

import openai
import pytest

from llm_backend import app


@pytest.fixture
def client():
    flask_app = app.create_app(logging.getLogger('server'))
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture(scope='session')
def set_openai_key():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(script_dir, '../../../openai_key.txt')
    assert os.path.exists(key_path), f"Please create 'openai_key.txt' at the root folder before executing the tests'"
    openai.api_key_path = key_path

@pytest.fixture(scope='session')
def set_logger():
    logger = logging.getLogger('server')
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)