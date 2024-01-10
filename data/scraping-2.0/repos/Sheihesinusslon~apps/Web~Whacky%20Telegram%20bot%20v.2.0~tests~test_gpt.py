import pytest
from openai import OpenAIError

from app.services import GPT
from app.config import SERVICE_ERROR_MSG
from tests.fakes import FakeOpenaiClient


@pytest.fixture
def setup_test_apps(monkeypatch):
    with monkeypatch.context():
        client = FakeOpenaiClient()
        gpt = GPT(openai_client=client)
        yield client, gpt
        client.reset()


class TestGPT:
    def test_init(self, setup_test_apps):
        client, gpt = setup_test_apps
        assert gpt.model_engine

    def test_generate_response(self, setup_test_apps):
        client, gpt = setup_test_apps
        prompt = "Prompt"
        response = gpt.generate_response(prompt)

        assert isinstance(response, str)
        assert client.called

    def test_generate_response_error(self, monkeypatch, setup_test_apps):
        def raise_exc(*a, **kw):
            raise OpenAIError()

        monkeypatch.setattr(FakeOpenaiClient, "create", raise_exc)
        client, gpt = setup_test_apps
        prompt = "Prompt"
        response = gpt.generate_response(prompt)

        assert response == SERVICE_ERROR_MSG
        assert not client.called

    def test_generate_random_response(self, setup_test_apps):
        client, gpt = setup_test_apps
        prompt = "Prompt"
        response = gpt.generate_random_response(prompt)

        assert isinstance(response, str)
        assert client.called

    def test_get_horoscope(self, setup_test_apps):
        client, gpt = setup_test_apps
        response = gpt.get_horoscope()

        assert isinstance(response, str)
        assert client.called
