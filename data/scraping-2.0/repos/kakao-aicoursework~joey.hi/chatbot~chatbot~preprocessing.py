from langchain.schema import Document


def doc_preprocessing(documents):
    processed_docs = list()
    page_content = documents[0].page_content
    metadata = documents[0].metadata

    idx = 0
    for char in page_content:
        if char == '#':
            idx += 1
            page_content = page_content[:idx] + " " + page_content[idx:]
        idx += 1
    idx = 1
    prev_idx = 0
    for char in page_content:
        if char == '#' and idx > 1:
            tmp = page_content[prev_idx:idx - 1]
            processed_docs.append(Document(page_content = tmp, metadata = metadata))
            prev_idx = idx - 1
        idx += 1
