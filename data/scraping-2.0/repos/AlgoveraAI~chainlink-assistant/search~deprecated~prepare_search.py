import re
import pickle
from tqdm import tqdm
from datetime import datetime
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from config import get_logger

logger = get_logger(__name__)


system_template = """
Please summarize the context below in one sentence (no more than 15 words). This will be used as the description of the article in the search results.

Response should be NO MORE THAN 15 words.
"""

human_template = """{context}"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
)

llm = ChatOpenAI(temperature=0.0)
chain = LLMChain(llm=llm, prompt=PROMPT)


def extract_first_n_paragraphs(content, num_para=2):

    # Split by two newline characters to denote paragraphs
    paragraphs = content.split("\n\n")

    # Return the first num_para paragraphs or whatever is available
    return "\n\n".join(paragraphs[:num_para])


def prepare_search_docs(doc_path, blog_path, num_para=2):
    with open(doc_path, "rb") as f:
        docs = pickle.load(f)

    with open(blog_path, "rb") as f:
        blogs = pickle.load(f)

    blog_docs = []
    for blog in tqdm(blogs, total=len(blogs)):
        title = blog.page_content.split("\n\n")[0].replace("#", "").strip()

        # Get the first two paragraphs
        para = extract_first_n_paragraphs(blog.page_content, num_para=num_para)

        description = chain.predict(context=para)

        metadata = {
            "title": title,
            "description": description,
            "source": blog.metadata["source"],
            "source_type": "blog",
        }
        logger.info(f"Description: {description}")

        blog.metadata = metadata
        blog_docs.append(blog)

    tech_docs = []
    for doc in tqdm(docs, total=len(docs)):
        title = doc.page_content.split("\n\n")[0].replace("#", "").strip()

        para = extract_first_n_paragraphs(doc.page_content, num_para=num_para)

        description = chain.predict(context=para)

        metadata = {
            "title": title,
            "description": description,
            "source": doc.metadata["source"],
            "source_type": "technical_document",
        }
        logger.info(f"Title: {title}")
        logger.info(f"Description: {description}")

        doc.metadata = metadata

        tech_docs.append(doc)

    # Save the documents
    with open(
        f"./data/search_blogdocs_{datetime.now().strftime('%Y-%m-%d')}.pkl", "wb"
    ) as f:
        pickle.dump(blog_docs, f)

    with open(
        f"./data/search_techdocs_{datetime.now().strftime('%Y-%m-%d')}.pkl", "wb"
    ) as f:
        pickle.dump(tech_docs, f)

    return blog_docs, tech_docs


if __name__ == "__main__":
    prepare_search_docs(
        doc_path="/home/marshath/play/chainlink/chainlink-assistant/data/techdocs_2023-08-14.pkl",
        blog_path="/home/marshath/play/chainlink/chainlink-assistant/data/blog_2023-08-14.pkl",
    )
