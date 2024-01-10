import logging,os
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"


def load_documents(SOURCE_DIRECTORY):
    # Check if the source directory exists
    if not os.path.exists(SOURCE_DIRECTORY):
        print(f"The source directory '{SOURCE_DIRECTORY}' does not exist.")
        return

    # List all files in the source directory
    document_files = os.listdir(SOURCE_DIRECTORY)

    # Filter only PDF files if needed
    pdf_files = [file for file in document_files if file.endswith('.pdf')]

    # Now you have a list of PDF files in the directory
    # You can perform further actions, like processing or loading these documents
    text = ""
    for pdf_file in pdf_files:
        # Do something with each PDF file, for example:
        # 1. Extract text from the PDF using pdfminer
        # 2. Store or process the extracted text
        # 3. Perform any other necessary operations

        # Example: Extract text using pdfminer
        # Replace 'your_pdf_file.pdf' with the actual path
        pdf_path = os.path.join(SOURCE_DIRECTORY, pdf_file)
        text+= extract_text(pdf_path)

        # Example: Print the extracted text


    return text

def main(device_type="cpu"):
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents_text = load_documents(SOURCE_DIRECTORY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=90)
    splited_text = text_splitter.create_documents([documents_text])
    print(splited_text)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )

    main()
