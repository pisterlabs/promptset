from lm4hpc._utils_langchain import (get_pdf_text, 
                                     get_chunk_text, 
                                     get_vector_store, 
                                     get_conversation_chain)
import os
import openai
import pytest

@pytest.fixture
def get_single_pdf_path(pdf_file = "lm4hpc/data/test/OMP_official_merged.pdf"):
    return pdf_file 

@pytest.fixture
def get_pdf_dir_path(pdf_dir = "/Users/lc/Documents/Code/LM4HPC/lm4hpc/data/test"):
    return pdf_dir

@pytest.fixture
def get_pdf_text_fixture(pdf_file = "lm4hpc/data/test/OMP_official_merged.pdf"):
    text = get_pdf_text(pdf_file)
    return text

@pytest.fixture
def get_pdf_chunks_fixture(pdf_file = "lm4hpc/data/test/OMP_official_merged.pdf"):
    text = get_pdf_text(pdf_file)
    chunks = get_chunk_text(text)
    return chunks

def test_get_pdf_text_single(get_single_pdf_path):
    """
    Test the get_pdf_text function.
    """
    # Test a single PDF file
    text = get_pdf_text(get_single_pdf_path)
    assert len(text) == 1455111

def test_get_pdf_text_dir(get_pdf_dir_path):
    """
    Test the get_pdf_text function.
    """
    # pdf_dir = os.path.join(os.path.dirname(__file__), 'data/openmp_pdfs')
    text = get_pdf_text(get_pdf_dir_path)
    assert len(text) == 1455111 * 2

def test_get_chunk_text(get_pdf_text_fixture):
    """
    Test the get_chunk_text function.
    """
    # text = get_pdf_text("lm4hpc/data/test/OMP_official_merged.pdf")
    chunks = get_chunk_text(get_pdf_text_fixture)
    assert len(chunks) == 1827

def test_get_vector_store(get_pdf_chunks_fixture):
    """
    Test the get_conversation_chain function.
    """
    # text = get_pdf_text("lm4hpc/data/test/OMP_official_merged.pdf")
    get_vector_store(get_pdf_chunks_fixture)

# if __name__ == '__main__':
#     test_get_pdf_text_single()
#     test_get_pdf_text_dir()
#     test_get_chunk_text()