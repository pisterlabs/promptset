
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def text_to_doc(text_arr):
    text = [text_arr]
    page_docs = [Document(page_content=page) for page in text]

    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    doc_chunksText = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1600, separators = ["\n\n", ".", "!", "?"], chunk_overlap = 0)
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={"page":doc.metadata["page"], "chunk": i})
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
            # doc_chunksText.append(chunk)
    return chunks