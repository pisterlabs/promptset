from unittest.mock import MagicMock

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.embedding import Embedding

from api.infra.db.model.file import FileChunk
from api.web.service.file_chunk import FileChunkService


@pytest.fixture
def file_chunk_service():
    file_chunk_repo_mock = MagicMock()
    openai_mock = MagicMock()
    return FileChunkService(
        file_chunk_repository=file_chunk_repo_mock,
        openai=openai_mock,
    )


def test_create_file_chunks_embedding_single_chunk(
    file_chunk_service: FileChunkService,
):
    file_chunk_service.num_tokens_from_string = MagicMock(return_value=400)
    file_chunk_service.openai.embeddings.create = MagicMock(
        return_value=CreateEmbeddingResponse(
            data=[Embedding(embedding=[1, 2, 3], index=0, object="embedding")],
            model="text-embedding-ada-002",
            object="list",
            usage={"prompt_tokens": 0, "total_tokens": 0},
        ),
    )
    file_chunk_service.file_chunk_repository.create = MagicMock()

    file_chunk_service.create_file_chunks_embedding(
        file_id=1,
        file_text_content="Test content",
    )

    file_chunk_service.num_tokens_from_string.assert_called_once_with("Test content")
    file_chunk_service.openai.embeddings.create.assert_called_once_with(
        model="text-embedding-ada-002",
        input="Test content",
    )
    expected_file_chunk = FileChunk(
        file_id=1,
        chunk_text="Test content",
        embedding_vector=[1, 2, 3],
    )
    # Extract the arguments passed to the create method
    actual_args, _ = file_chunk_service.file_chunk_repository.create.call_args
    # Check if the arguments match the expected FileChunk object
    assert actual_args[0].file_id == expected_file_chunk.file_id
    assert actual_args[0].chunk_text == expected_file_chunk.chunk_text
    assert actual_args[0].embedding_vector == expected_file_chunk.embedding_vector


def test_create_file_chunks_embedding_multiple_chunks(
    file_chunk_service: FileChunkService,
):
    # Mocking the behavior for a larger text content that will be split into multiple chunks
    file_chunk_service.num_tokens_from_string = MagicMock(return_value=600)
    file_chunk_service.split_text_into_chunks = MagicMock(
        return_value=["Chunk 1", "Chunk 2"],
    )
    file_chunk_service.openai.embeddings.create = MagicMock(
        side_effect=[
            CreateEmbeddingResponse(
                data=[Embedding(embedding=[1, 2, 3], index=0, object="embedding")],
                model="text-embedding-ada-002",
                object="list",
                usage={"prompt_tokens": 0, "total_tokens": 0},
            ),
            CreateEmbeddingResponse(
                data=[Embedding(embedding=[4, 5, 6], index=0, object="embedding")],
                model="text-embedding-ada-002",
                object="list",
                usage={"prompt_tokens": 0, "total_tokens": 0},
            ),
        ],
    )
    file_chunk_service.file_chunk_repository.create = MagicMock()

    file_chunk_service.create_file_chunks_embedding(
        file_id=1,
        file_text_content="Large test content to split",
    )

    file_chunk_service.num_tokens_from_string.assert_called_once_with(
        "Large test content to split",
    )
    file_chunk_service.split_text_into_chunks.assert_called_once_with(
        "Large test content to split",
    )
    file_chunk_service.openai.embeddings.create.assert_any_call(
        model="text-embedding-ada-002",
        input="Chunk 1",
    )
    file_chunk_service.openai.embeddings.create.assert_any_call(
        model="text-embedding-ada-002",
        input="Chunk 2",
    )
    assert file_chunk_service.openai.embeddings.create.call_count == 2

    actual_calls = file_chunk_service.file_chunk_repository.create.call_args_list
    # Create a list of dictionaries representing expected calls
    expected_calls = [
        {"file_id": 1, "chunk_text": "Chunk 1", "embedding_vector": [1, 2, 3]},
        {"file_id": 1, "chunk_text": "Chunk 2", "embedding_vector": [4, 5, 6]},
    ]

    # Convert actual calls to dictionaries for comparison
    actual_calls_dict = [
        {
            "file_id": call_args[0][0].file_id,
            "chunk_text": call_args[0][0].chunk_text,
            "embedding_vector": call_args[0][0].embedding_vector,
        }
        for call_args in actual_calls
    ]

    # Convert expected calls and check if all expected calls are found in the actual calls
    for expected_call in expected_calls:
        assert (
            expected_call in actual_calls_dict
        ), f"Expected call not found: {expected_call}"
    assert len(actual_calls_dict) == len(expected_calls), "Unexpected number of calls"


def test_split_text_into_chunks():
    test_text = "This is a test sentence for chunk splitting. " * 600
    # 8 words times 600 repetitions = 4800 words
    # 4800 words divided by 384 words as ideal size = 13 chunks
    actual_chunks = FileChunkService.split_text_into_chunks(test_text)
    assert len(actual_chunks) == 13


def test_num_tokens_from_string():
    test_string = "This is a test file content for embedding cost calculation"
    expected_token_count = 10
    actual_token_count = FileChunkService.num_tokens_from_string(test_string)
    assert actual_token_count == expected_token_count


def test_get_file_words_length():
    test_file_content = "This is a test file content with several words"
    expected_word_count = 9
    actual_word_count = FileChunkService.get_file_words_length(test_file_content)
    assert actual_word_count == expected_word_count


def test_calculate_embedding_cost():
    test_file_content = (
        "This is a test file content for embedding cost calculation" * 1000
    )
    # 10 tokens times 1000 repetitions = 10000 tokens = 0.001 dollars
    expected_cost = 0.001
    actual_cost = FileChunkService.calculate_embedding_cost(test_file_content)
    # Assert approximately equal due to floating point comparison
    assert round(actual_cost, 4) == expected_cost
