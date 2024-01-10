from langchain.docstore.document import Document
from langchain.vectorstores.docarray import DocArray
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_docarray() -> None:
    texts = ['foo', 'bar', 'baz']
    docsearch = DocArray.from_texts(texts, FakeEmbeddings())
    output = docsearch.similarity_search('foo', k=1)
    assert output == [Document(page_content='foo')]


def test_docarray_with_scores() -> None:
    texts = ['foo', 'bar', 'baz']
    metadatas = [{'page': i} for i in range(len(texts))]
    docsearch = DocArray.from_texts(texts, FakeEmbeddings(), metadatas=metadatas)
    output = docsearch.similarity_search_with_score('foo', k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == list([Document(page_content=t, metadata={'page': idx}) for idx, t in enumerate(texts)])
    assert scores[0] > scores[1] > scores[2]
