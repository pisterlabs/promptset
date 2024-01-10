from langchain.schema import Document
from onepoint_document_chat.service.text_extraction import (
    PAGE,
    FILE_NAME,
    merge_docs,
    combine_documents,
)
from onepoint_document_chat.service.test.doc_provider import (
    create_doc1,
    create_doc2,
    create_list_simple1,
    create_single_long_doc_list,
    create_list_3,
)


def test_merge_docs():
    doc1 = create_doc1()
    doc2 = create_doc2()
    merged = merge_docs(doc1, doc2)
    assert merged is not None
    assert merged.page_content is not None
    assert merged.page_content == "This is doc1\nThis is doc2"
    assert merged.metadata[PAGE] == [1, 2]
    assert merged.metadata[FILE_NAME] == "file1, file1"


def test_combine_documents():
    docs = create_list_simple1()
    combined_docs = combine_documents(docs)
    assert len(combined_docs) == 1
    assert combined_docs[0].metadata[FILE_NAME] == "file1, file1"
    assert combined_docs[0].page_content == "This is doc1\nThis is doc2"


def test_combine_single_long_doc():
    docs = create_single_long_doc_list()
    combined_docs = combine_documents(docs)
    assert len(combined_docs) == 1
    combined_docs[0].metadata[FILE_NAME] == "file_long_1"


def test_combine_3():
    docs = create_list_3()
    combined_docs = combine_documents(docs)
    assert len(combined_docs) == 1
