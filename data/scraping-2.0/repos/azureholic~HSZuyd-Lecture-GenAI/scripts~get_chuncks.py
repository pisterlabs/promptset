from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import MarkdownHeaderTextSplitter

loader = AsyncChromiumLoader(["https://www.zuyd.nl/opleidingen/hbo-ict"])
html = loader.load()

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(html)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(docs_transformed[0].page_content)

for (i, split) in enumerate(md_header_splits):
    file_object = open("chunks/hbo-ict-" + str(i) + ".md", "w")
    file_object.write(split.page_content)
    file_object.close()
