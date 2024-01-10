from vectorstore import count, delete_store, save_documents, similarity_query
from langchain.schema.document import Document


def test_delete_store_resets_the_db():
    delete_store()
    save_documents([Document(page_content="After delete document")])


def test_round_trip() -> None:
    initial_count = count()
    test_document = Document(page_content="A document")
    save_documents([test_document])
    docs = similarity_query("What is a document?")
    assert len(docs) == initial_count + 1
    assert docs[0] == test_document
