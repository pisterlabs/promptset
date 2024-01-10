import numpy as np
import pytest

from python.embeddings.embedding import generate_embedding
from python.embeddings.openai_embedder import OpenAIEmbedder


@pytest.mark.vcr
def test_embedding():
    e = generate_embedding("hello world", embedder=OpenAIEmbedder)

    # TODO no idea what an effective test here is...
    assert isinstance(e, np.ndarray)
    assert len(e) > 1000
