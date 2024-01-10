import pytest
from unittest.mock import patch
from app.utils import langchain_utils
from app.service.knowledgebase_handler import set_knowledge_base

@pytest.fixture(autouse=True)
def reset_context():
    set_knowledge_base(None)  # Reset the context to an empty string
    yield  # Yield to allow the test to run
    set_knowledge_base(None)  # Reset the context again after the test completes

def test_process_file_context_exception():
    """
    Test that generate_chatgpt_response function raises an exception
    when the OpenAI API call fails.
    """
    with patch('os.getenv', return_value='testkey'):
        # Mock OpenAI API call to raise an exception
        with patch("langchain.vectorstores.FAISS.from_texts", side_effect=Exception("Unexpected error")):
            with pytest.raises(Exception) as e_info:
                langchain_utils.process_file_context("test text")
            assert "An error occurred during process_file_context()" in str(e_info.value)

def test_process_user_question_exception():
    """
    Test that generate_chatgpt_response function raises an exception
    when the OpenAI API call fails.
    """
    with patch('os.getenv', return_value='testkey'):
        # Mock OpenAI API call to raise an exception
        with patch("langchain.chains.question_answering.load_qa_chain", side_effect=Exception("Unexpected error")):
            with pytest.raises(Exception) as e_info:
                langchain_utils.process_user_question("test text")
            assert "An error occurred during process_user_question()" in str(e_info.value)
