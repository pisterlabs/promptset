from langchain.docstore.document import Document
from langchain.document_transformers import LongContextReorder


def lost_in_the_middle(documents: list[Document]):
    reordering = LongContextReorder()
    reordered_documents = reordering.transform_documents(documents=documents)

    return reordered_documents
