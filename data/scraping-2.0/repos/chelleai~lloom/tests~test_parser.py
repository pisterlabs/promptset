from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from lloom import Migration
from lloom.dataset.textfile_dataset import TextfileDataset


def test_migration():
    m = Migration(file_path="tests/sotu-small.yml")
    m.run_migration()

    assert isinstance(m.datasets["sotu_raw"], TextfileDataset)
    assert m.stores["sotu_db"].count() == 1
    coll = m.stores["sotu_db"]
    assert isinstance(coll._embedding_function, OpenAIEmbeddingFunction)
