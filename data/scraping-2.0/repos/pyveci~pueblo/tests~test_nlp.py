# TODO: Publish as real package.
from pueblo.testing.nlp import nltk_init  # noqa: F401


def test_cached_web_resource():
    from pueblo.nlp.resource import CachedWebResource

    url = "https://github.com/langchain-ai/langchain/raw/v0.0.325/docs/docs/modules/state_of_the_union.txt"
    docs = CachedWebResource(url).langchain_documents(chunk_size=1000, chunk_overlap=0)
    assert len(docs) == 42

    from langchain.schema import Document

    assert isinstance(docs[0], Document)


def test_nltk_init(nltk_init):  # noqa: F811
    """
    Just _use_ the fixture to check if it works well.

    TODO: Anything else that could be verified here?
    """
    pass
