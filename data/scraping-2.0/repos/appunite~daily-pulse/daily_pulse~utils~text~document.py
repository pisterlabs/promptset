from langchain.docstore.document import Document


def mapTextsToDocuments(texts):
    return [Document(page_content=t) for t in texts]
