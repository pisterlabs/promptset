from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

def file_preprocessing(file, chunk_size=200, chunk_overlap=50, verbose=False):
    """
    Loads a PDF file, splits it into chunks, and concatenates these chunks.

    Parameters:
    file (str): Path to the PDF file.
    chunk_size (int): The size of each chunk.
    chunk_overlap (int): The overlap between chunks.
    verbose (bool): If True, prints each chunk.

    Returns:
    str: The concatenated text from the PDF.
    """
    try:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(pages)

        if verbose:
            for text in texts:
                print(text.page_content)

        final_texts = ''.join([text.page_content for text in texts])
        return final_texts
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None