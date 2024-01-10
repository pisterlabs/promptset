import pytest

from clients import OpenaiClient


@pytest.fixture
def open_ai_client() -> OpenaiClient:
    return OpenaiClient(token="some-token-top-secret")
