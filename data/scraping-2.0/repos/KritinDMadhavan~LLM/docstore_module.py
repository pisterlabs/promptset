
from langchain.docstore.document import Document

def document_splitter(tables):
    documents = []
    for t in tables:
        flattened_metadata = {
            key: value
            for key, value in t['schema'].items()
            if isinstance(value, (str, int, float, bool))
        }

        params = {
            "page_content": f"Table: {t['name']}",
            "metadata": flattened_metadata
        }

        document = Document(**params)
        documents.append(document)

    return documents
