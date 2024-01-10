
from langchain.document_loaders.sitemap import SitemapLoader

loader = SitemapLoader(
    "https://giki.edu.pk/sitemap_index.xml",
     )
docs = loader.load()


from  langchain.schema import Document
from typing import Iterable

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

save_docs_to_jsonl(docs,"data.jsonl")