# romeo-gtp/romeo_gpt/utils.py
import openai
import pandas as pd
import warnings
from romeo_gpt.utils.models.models import get_embedding
from langchain.text_splitter import TokenTextSplitter

from romeo_gpt.utils.connectors.extract_pdf import extract_text_from_pdf
from romeo_gpt.utils.connectors.extract_docx import extract_text_from_docx
from romeo_gpt.utils.connectors.extract_txt import extract_text_from_txt


def extract_text_from_files(df, file_contents) -> pd.DataFrame:
    for file_content, file_extension in file_contents:
        if file_extension == "pdf":
            text = extract_text_from_pdf(file_content)
        elif file_extension == "docx":
            text = extract_text_from_docx(file_content)
        elif file_extension == "txt":
            text = extract_text_from_txt(file_content)
        else:
            text = b""
            warnings.warn(f"Unsupported file format: {file_extension}")
        new_row = pd.DataFrame(
            {"document_name": [file_content], "text_extracted": [text]}
        )
        df = pd.concat([df, new_row], ignore_index=True)
    return df


def encode_text(text):
    if text is None or len(text) == 0:
        return None
    return text.decode("utf-8").replace("\n", " ")


def split_text_chunks(df) -> pd.DataFrame:
    text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
    df["text_chunks"] = ""

    for index, row in df.iterrows():
        document_name = row["document_name"]
        text_extracted = row["text_extracted"]
        print(f"Document: {document_name}, Text Extracted: {text_extracted}")
        if isinstance(text_extracted, bytes):
            text_extracted = text_extracted.decode("utf-8")
        text_chunks = text_splitter.split_text(text_extracted)

        comma_separated_chunks = ", ".join(text_chunks)

        df.at[index, "text_chunks"] = comma_separated_chunks

    return df


def intermediate_processor(file_contents: list) -> pd.DataFrame:
    df = pd.DataFrame({"document_name": [], "text_extracted": [], "text_chunks": []})
    df = df.pipe(extract_text_from_files, file_contents).pipe(split_text_chunks)
    return df


def primary_processor(df, api_key: str) -> pd.DataFrame:
    openai.api_key = api_key

    def embed_text(text, model="text-embedding-ada-002"):
        return get_embedding(text, api_key, model=model)

    def assign_vector_id():
        return list(range(len(df)))

    df["text_embeddings"] = df["text_chunks"].apply(
        lambda x: embed_text(x, model="text-embedding-ada-002")
    )
    df["document_name_embeddings"] = df["document_name"].apply(
        lambda x: embed_text(x, model="text-embedding-ada-002")
    )
    df["vector_id"] = assign_vector_id()
    return df
