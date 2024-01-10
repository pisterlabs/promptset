from langchain.schema import Document
from typing import List


def printdocs(resp: List[Document]):
    print(f"Found {len(resp)} docs")

    i = 0
    for r in resp:
        d = r
        print("-------")
        print(f"Document {i}:")
        print("-------")
        print("Metadata:")
        print(f"Domain: {d.metadata['domain']}")
        print(f"Title: {d.metadata['title']}")
        print(f"URL: {d.metadata['url']}")
        print("")
        print(d.page_content)
        print("")
        i = i + 1
