from hashlib import md5

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from loguru import logger
from newspaper import Article


def download(url):
    logger.info(f"Downloading article {url}...")
    article = Article(url)
    article.download()
    article.parse()
    return article.text


def prepare_documents(data, parser):
    sources = []
    for row in data:
        content = download(row["claimReview"][0]["url"])
        content = parser(content)
        _hash = md5(row["claimReview"][0]["title"].encode()).hexdigest()
        sources.append(
            Document(
                page_content=content,
                metadata={
                    "url": row["claimReview"][0]["url"],
                    "hash": _hash,
                    "review_date": row["claimReview"][0]["reviewDate"]
                }
            )
        )

    separator = r"\."
    splitter = CharacterTextSplitter(separator, chunk_size=2048, chunk_overlap=0)
    documents = splitter.split_documents(sources)

    for doc in documents:
        doc.page_content = doc.page_content.replace(separator, ".")

    return documents
