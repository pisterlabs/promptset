from typing import List

import requests
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def split_documents(documents,
                    verbose=False
                    ) -> List[Document]:
    clean_documents = [replace_non_ascii(doc) for doc in documents]

    # The default list of separators is ["\n\n", "\n", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
        length_function=len,
        separators=["---", "####", "####", "##", "#", "\n\n\n", "\n\n", "\n", ".", "?", "!", " ", ""],
    )

    document_chunks = text_splitter.split_documents(clean_documents)

    if verbose:
        print("\n")
        for doc in document_chunks:
            print("------------------------")
            print(f"{doc.page_content}\n")
    return document_chunks


def replace_non_ascii(doc: Document) -> Document:
    """
    Replaces non-ascii characters with ascii characters
    """
    page_content = doc.page_content \
        .replace("\ue05c", "fi") \
        .replace("\ufb01", "fi") \
        .replace("\x00", " ") \
        .replace('\u0000', " ")
    page_content_ascii = page_content.encode("ascii", "ignore").decode()
    return Document(page_content=page_content_ascii, metadata=doc.metadata)


def ntfy_notification(message: str,
                      title='Successful',
                      tags='info'
                      ):
    requests.post("https://ntfy.sh/johntday_alerts",
                  data=message,
                  headers={
                      "Title": title,
                      "Priority": "urgent",
                      "Tags": tags
                  })
