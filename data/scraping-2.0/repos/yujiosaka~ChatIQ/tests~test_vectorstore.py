import pytest
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores.weaviate import Weaviate
from weaviate import Client
from weaviate.batch import Batch
from weaviate.schema import Schema

from chatiq.retriever import Retriever
from chatiq.vectorstore import Vectorstore


@pytest.fixture
def mock_weaviate_client(mocker):
    mock_weaviate_client = mocker.MagicMock(spec=Client)
    mock_batch = mocker.MagicMock(spec=Batch)
    mock_batch.__enter__.return_value = mock_batch
    mock_batch.__exit__.return_value = None
    mock_weaviate_client.batch = mock_batch
    mock_weaviate_client.schema = mocker.MagicMock(spec=Schema)
    return mock_weaviate_client


@pytest.fixture
def mock_weaviate(mocker):
    mock_weaviate = mocker.MagicMock(spec=Weaviate)
    mock_weaviate.as_retriever.return_value = mocker.MagicMock(spec=VectorStoreRetriever)
    mocker.patch("chatiq.vectorstore.Weaviate", return_value=mock_weaviate)
    return mock_weaviate


@pytest.fixture
def mock_document(mocker):
    return mocker.MagicMock(spec=Document)


@pytest.fixture
def mock_retriever_init(mocker):
    return mocker.patch.object(Retriever, "__init__", return_value=None)


@pytest.mark.parametrize("docs", [[], [mock_document], [mock_document, mock_document]])
def test_add_documents(mock_weaviate_client, mock_weaviate, mock_document, docs):
    vectorstore = Vectorstore(mock_weaviate_client, "team_id1")
    vectorstore.add_documents(docs)
    mock_weaviate.add_documents.assert_called_once_with(docs)


def test_delete_channel(mock_weaviate_client, mock_weaviate):
    vectorstore = Vectorstore(mock_weaviate_client, "team_id1")
    vectorstore.delete_channel("T0JD6RZU6")
    mock_weaviate_client.batch.delete_objects.assert_called_once_with(
        class_name="Messageteam_id1", where={"path": ["channel_id"], "operator": "Equal", "valueString": "T0JD6RZU6"}
    )


def test_delete_message(mock_weaviate_client, mock_weaviate):
    vectorstore = Vectorstore(mock_weaviate_client, "team_id1")
    vectorstore.delete_message("1629470261.000200")
    mock_weaviate_client.batch.delete_objects.assert_called_once_with(
        class_name="Messageteam_id1", where={"path": ["ts"], "operator": "Equal", "valueString": "1629470261.000200"}
    )


def test_delete_file_or_attachment(mock_weaviate_client, mock_weaviate):
    vectorstore = Vectorstore(mock_weaviate_client, "team_id1")
    vectorstore.delete_file_or_attachment("1629470261.000200-1")
    mock_weaviate_client.batch.delete_objects.assert_called_once_with(
        class_name="Messageteam_id1",
        where={"path": ["file_or_attachment_id"], "operator": "Equal", "valueString": "1629470261.000200-1"},
    )


@pytest.mark.parametrize("index_exists", [True, False])
def test_ensure_index(mock_weaviate_client, mock_weaviate, index_exists):
    mock_weaviate_client.schema.exists.return_value = index_exists

    vectorstore = Vectorstore(mock_weaviate_client, "team_id1")
    vectorstore.ensure_index()

    if index_exists:
        mock_weaviate_client.schema.create.assert_not_called()
        mock_weaviate.add_documents.assert_not_called()
    else:
        mock_weaviate_client.schema.create.assert_called_once()
        mock_weaviate.add_documents.assert_called_once()


@pytest.mark.parametrize("index_exists", [True, False])
def test_delete_index(mock_weaviate_client, mock_weaviate, index_exists):
    mock_weaviate_client.schema.exists.return_value = index_exists

    vectorstore = Vectorstore(mock_weaviate_client, "team_id1")
    vectorstore.delete_index()

    if index_exists:
        mock_weaviate_client.schema.delete_class.assert_called_once()
    else:
        mock_weaviate_client.schema.delete_class.assert_not_called()


def test_as_retriever(mock_weaviate_client, mock_weaviate, mock_retriever_init):
    vectorstore = Vectorstore(mock_weaviate_client, "team_id1")
    vectorstore.as_retriever(True, "T0JD6RZU6", "1629470261.000200")
    mock_retriever_init.assert_called_once_with(mock_weaviate, True, "T0JD6RZU6", "1629470261.000200")
