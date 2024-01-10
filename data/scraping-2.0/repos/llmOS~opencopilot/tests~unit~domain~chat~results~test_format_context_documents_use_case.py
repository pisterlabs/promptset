from langchain.schema import Document

from opencopilot.domain.chat.results import format_context_documents_use_case as use_case


def test_basic():
    documents = [
        Document(
            page_content="mock content",
            metadata={
                "title": "mock title",
                "source": "mock source"
            }
        )
    ]
    result = use_case.execute(documents)
    assert result == "Title: mock title\nSource: mock source\nContent:\nmock content"


def test_basic_multiple():
    documents = [
        Document(
            page_content="mock content",
            metadata={
                "title": "mock title",
                "source": "mock source"
            }
        ),
        Document(
            page_content="mock content2",
            metadata={
                "title": "mock title2",
                "source": "mock source2"
            }
        ),
    ]
    result = use_case.execute(documents)
    assert result == "Title: mock title\nSource: mock source\nContent:\nmock content\n\nTitle: mock title2\nSource: mock source2\nContent:\nmock content2"


def test_missing_title():
    documents = [
        Document(
            page_content="mock content",
            metadata={
                "source": "mock source"
            }
        )
    ]
    result = use_case.execute(documents)
    assert result == "Source: mock source\nContent:\nmock content"


def test_missing_source():
    documents = [
        Document(
            page_content="mock content",
            metadata={
                "title": "mock title"
            }
        )
    ]
    result = use_case.execute(documents)
    assert result == "Title: mock title\nContent:\nmock content"


