from discontinuity_api.vector import get_vector_db, add_document, query_index
from langchain.docstore.document import Document


def test_get_vector_db():
    vectorIndex = get_vector_db("testing")

    assert vectorIndex is not None


def test_add_document_to_vector_db():
    vectorIndex = get_vector_db("testing")
    testDocument = Document(page_content="Oranges are great fruits")

    add_document(index=vectorIndex, documents=[testDocument])

    response = query_index(index=vectorIndex, query="Are bananas vegetables?")

    print("\nResponse:", response)

    assert response is not None

    # The response shouldn't have "the context does not"
    assert "the context does not" not in response
