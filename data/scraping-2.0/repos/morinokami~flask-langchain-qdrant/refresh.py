import os
import shutil

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from pypdf import PdfReader

from const import QDRANT_PATH, COLLECTION_NAME, DOCUMENTS_PATH


def split(text: str):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=250,
        chunk_overlap=0,
    )
    return splitter.split_text(text)


def extract_text(pdf_path: str):
    reader = PdfReader(pdf_path)
    return "\n\n".join([page.extract_text() for page in reader.pages])


def main():
    if os.path.exists(QDRANT_PATH):
        shutil.rmtree(QDRANT_PATH)

    pdfs = [
        os.path.join("documents", pdf)
        for pdf in os.listdir(DOCUMENTS_PATH)
        if pdf.endswith(".pdf")
    ]
    if not pdfs:
        raise Exception("No PDFs found in ./documents")
    for pdf in pdfs:
        text = extract_text(pdf)
        split_text = split(text)
        if text:
            print(f"Adding {pdf} to Qdrant")
            Qdrant.from_texts(
                split_text,
                OpenAIEmbeddings(),
                path=QDRANT_PATH,
                collection_name=COLLECTION_NAME,
            )


if __name__ == "__main__":
    main()
