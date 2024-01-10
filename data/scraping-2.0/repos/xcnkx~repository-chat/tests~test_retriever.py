from app.retriver import create_retriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document


def test_create_retriever():
    # Create some dummy data
    embeddings = OpenAIEmbeddings()
    splits = [
        Document(
            page_content="This is the first document.",
            metadata={"title": "first document"},
        ),
        Document(
            page_content="This is the second document.",
            metadata={"title": "second document"},
        ),
        Document(
            page_content="This is the third document.",
            metadata={"title": "third document"},
        ),
    ]

    # Call the function
    retriever = create_retriever(embeddings, splits)

    # Check that the retriever returns the expected results
    query = "first document"
    results = retriever.get_relevant_documents(query)
    assert len(results) == 3
    assert results[0].page_content == "This is the first document."

    query = "second document"
    results = retriever.get_relevant_documents(query)
    assert len(results) == 3
    assert results[0].page_content == "This is the second document."

    query = "third document"
    results = retriever.get_relevant_documents(query)
    assert len(results) == 3
    assert results[0].page_content == "This is the third document."
