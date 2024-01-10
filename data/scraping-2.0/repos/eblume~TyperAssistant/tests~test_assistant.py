"""Tests of assistant code."""

import openai
import pytest
import typer
from typerassistant.assistant import Assistant, RemoteAssistant, Thread
from typerassistant.typer import TyperAssistant


@pytest.fixture
def mock_remote_assistant(mocker):
    remote = mocker.MagicMock(spec=RemoteAssistant)
    remote.id = "test assistant id"
    remote.name = "test assistant name"
    remote.instructions = "test instructions"
    return remote


@pytest.fixture
def mock_thread(mocker):
    thread = mocker.MagicMock(spec=Thread)
    thread.id = "test thread id"
    return thread


@pytest.fixture
def mock_client(mocker, mock_thread):
    client = mocker.MagicMock(spec=openai.OpenAI)
    client.beta = mocker.MagicMock()  # mocking this API is going to be a nightmare
    client.beta.assistants.delete.return_value = None
    client.beta.threads.retrieve.return_value = mock_thread
    return client


@pytest.fixture(params=[Assistant, TyperAssistant])
def assistant_class(request):
    """Return the Assistant class to test."""
    cls = request.param
    return cls


@pytest.fixture
def typer_app():
    """An example Typer app."""
    app = typer.Typer(name="test typer app")

    @app.command()
    def say_hello(name: str):
        print(f"Hello, {name}")

    return app


@pytest.fixture
def assistant(assistant_class, typer_app, mock_client, mock_remote_assistant):
    """Return an assistant instance."""
    if assistant_class == TyperAssistant:
        return TyperAssistant(app=typer_app, client=mock_client, _assistant=mock_remote_assistant)
    return assistant_class(name=mock_remote_assistant.name, client=mock_client, _assistant=mock_remote_assistant)


@pytest.fixture
def saved_assistant(assistant, mock_client, mock_remote_assistant):
    """An assistant that can be retrieved from the API"""
    mock_client.beta.assistants.retrieve.return_value = mock_remote_assistant
    return assistant


def test_load_from_id(saved_assistant, typer_app, mock_client):
    """Test loading an assistant from its ID."""
    # Note: Unfortunately due to an issue with typing I ran in to (explained in typer.py's TyperAssistant.from_id
    # method), we have to use different from_id methods for different Assistant subclasses. This makes all the work I
    # did setting up fixtures for this test a bit pointless, but I'm leaving it in for now in case I want to use it in
    # the future.
    if isinstance(saved_assistant, TyperAssistant):
        loaded = TyperAssistant.from_id_with_app(saved_assistant.assistant.id, typer_app, client=mock_client)
    else:
        loaded = saved_assistant.__class__.from_id(saved_assistant.assistant.id, client=mock_client)
    assert loaded.assistant.id == saved_assistant.assistant.id
    assert loaded.name == saved_assistant.name
