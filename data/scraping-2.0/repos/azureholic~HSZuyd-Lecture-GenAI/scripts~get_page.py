from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import MarkdownHeaderTextSplitter

pagesToLoad = [
    "https://www.zuyd.nl/",
    "https://www.zuyd.nl/opleidingen/hbo-ict",
    "https://www.zuyd.nl/opleidingen/applied-data-science-and-artificial-intelligence"
]


loader = AsyncChromiumLoader(pagesToLoad)
html = loader.load()

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(html)

for (i, doc) in enumerate(docs_transformed):
    file_object = open(f"docs/page-{str(i)}.md", "w")
    file_object.write(doc.page_content)
    file_object.close()

