from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

if __name__ == '__main__':

    # Specify the directory path where you want to search for PDF files
    directory_path = Path(__file__).parent / "data" / "raw"

    # Use the glob module to search for PDF files in the specified directory
    pdf_files = list(directory_path.glob('*.pdf'))
    langchain_documents = []
    # Print the list of PDF files found
    for document in pdf_files:
        loader = PyPDFLoader(document)
        data = loader.load()
        langchain_documents.extend(data)

    print(langchain_documents)
