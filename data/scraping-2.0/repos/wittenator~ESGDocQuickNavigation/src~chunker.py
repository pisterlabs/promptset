from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Extracting Text from PDF
def extract_text_from_pdf(file_path):
    texts = []
    metadata = []
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        for i, page in enumerate(pdf.pages):
            texts.append(page.extract_text())
            metadata.append({"page": i+1})
    return texts, metadata


def get_splits(size, overlap, text, metadata):
    # Initialize the text splitter with custom parameters
    custom_text_splitter = RecursiveCharacterTextSplitter(
        # Set custom chunk size
        chunk_size = size,
        chunk_overlap  = overlap,
        # Use length of the text as the size measure
        length_function = len,

    )
    return custom_text_splitter.create_documents(text, metadata)


if __name__=="__main__":
    # Extract text from the PDF and split it into sentences
    sample = extract_text_from_pdf("./data/Brazil.pdf")
    texts = get_splits(200, 20, sample)
    print(f"Sample: {texts[5]}")

