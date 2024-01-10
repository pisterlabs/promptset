from pytest import raises
from fastapi import HTTPException
from unittest.mock import patch, MagicMock
from app.controller import controller_document
from langchain.schema import Document


class TestDocumentController:
    def test_add_document_should_raise_exception(self):
        with raises(HTTPException) as exc_info:
            controller_document.add_document("new_database", "query")
        assert isinstance(exc_info.value, HTTPException)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "knowledge base not found"

    @patch("app.controller.controller_document.FAISS")
    def test_add_document_should_initialize_database(self, mock_FAISS):
        mocked_created_databases = {"new_database": None}
        with patch.dict(
            "app.databases_holder.database_container", mocked_created_databases
        ) as new_dict:
            mock_db = MagicMock()
            mock_FAISS.from_documents.return_value = mock_db
            controller_document.add_document("new_database", "query")
            assert mock_db == new_dict["new_database"]

    def test_add_document_should_add_documento_to_database(self):
        mock_database = MagicMock()
        mock_database.add_documents
        mocked_created_databases = {"new_database": mock_database}

        with patch.dict(
            "app.databases_holder.database_container", mocked_created_databases
        ):
            controller_document.add_document("new_database", "query")
        mock_database.add_documents.assert_called_once_with(
            [Document(page_content="query", metadata={"source": "local"})]
        )

    def test_delete_document_should_raise_db_not_found_exception(self):
        with raises(HTTPException) as exc_info:
            controller_document.delete_document("new_database", "index")
        assert isinstance(exc_info.value, HTTPException)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "knowledge base not found"

    def test_delete_document_should_raise_document_not_found_exception(self):
        mock_db = MagicMock()
        mock_db.delete.side_effect = ValueError()
        mocked_created_databases = {"new_database": mock_db}
        with patch.dict(
            "app.databases_holder.database_container", mocked_created_databases
        ):
            with raises(HTTPException) as exc_info:
                controller_document.delete_document("new_database", "query")
            assert isinstance(exc_info.value, HTTPException)
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail == "Value does not exist on database"

    def test_delete_document_should_delete_document(self):
        mock_database = MagicMock()
        mocked_created_databases = {"new_database": mock_database}

        with patch.dict(
            "app.databases_holder.database_container", mocked_created_databases
        ):
            controller_document.delete_document("new_database", "query")
        mock_database.delete.assert_called_once_with(["query"])
