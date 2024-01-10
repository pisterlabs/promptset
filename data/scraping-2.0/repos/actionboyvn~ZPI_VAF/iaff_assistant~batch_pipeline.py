from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
import re
import TranslationAgent
import asyncio

DOC_REPO_DIRECTORY = "documents/"
HF_EMBEDDINGS = 'sentence-transformers/msmarco-distilbert-base-v4'
CHROMA_LOCAL = "chroma_db"
LANGUAGES = ["English", "Polish", "Vietnamese", "Belarusian", "Ukrainian", "Russian"]
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDINGS)
vector_store = Chroma(persist_directory=CHROMA_LOCAL, embedding_function=embeddings,
                      collection_metadata={"hnsw:space": "cosine"})


async def preprocess_docs():
    doc_paths = []

    w = os.walk(DOC_REPO_DIRECTORY)
    for (path, name, file_name) in w:
        for file in file_name:
            try:
                df = pd.read_excel(path + "/" + file)
                if set(['short_description', 'info', 'useful_links']).issubset(df.columns):
                    doc_paths.append(path + "/" + file)
            except Exception:
                pass

    for doc_path in doc_paths:
        await ingest_doc(doc_path)


def get_links_from_text(text):
    pattern = r'(https://\S+)'

    matches = re.findall(pattern, text)

    if matches:
        return matches
    else:
        return "None"


async def ingest_doc(doc_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=50
    )
    df = pd.read_excel(doc_path)

    links = df['useful_links'][0]
    article = df['title'][0]
    content = df['info'][0]

    chunks = text_splitter.split_text(text=content)
    for i, chunk in enumerate(chunks):
        metadata = {"source": links,
                    "article": article,
                    "English": chunk}
        tasks = []
        for lang in LANGUAGES:
            if lang != "English":
                task = asyncio.create_task(TranslationAgent.translate(chunk, lang))
                tasks.append(task)

        translations = await asyncio.gather(*tasks)

        for lang, translation in zip([l for l in LANGUAGES if l != "English"], translations):
            metadata[lang] = translation

        new_doc = Document(
            page_content=chunk,
            metadata=metadata
        )
        vector_store.add_documents([new_doc], ids=[doc_path + "_chunk" + str(i)])


if __name__ == '__main__':
    asyncio.run(preprocess_docs())
