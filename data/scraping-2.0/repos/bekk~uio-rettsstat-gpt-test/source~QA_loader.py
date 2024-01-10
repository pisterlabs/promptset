from langchain.document_loaders import (
    DirectoryLoader,
    UnstructuredPDFLoader,
)
import re


def extract_text_from_pdf(dirictory):
    loader = DirectoryLoader(
        dirictory, glob="*.pdf", loader_cls=UnstructuredPDFLoader
    ).load()

    query_list = []

    for doc in loader:
        pattern = r"(Klager har i det vesentlige anført.*?)Tjenesteyterne har i det vesentlige anført"
        match = re.search(pattern, doc.page_content, re.DOTALL)
        full_text = match.group(1).strip()
        query_list.append(full_text)
    return query_list


def main():
    dir_path = "QApdf"
    print(extract_text_from_pdf(dir_path))


if __name__ == "__main__":
    main()
