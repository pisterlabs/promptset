import pytest
from dotenv import load_dotenv
from langchain import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import WebResearchRetriever
from langchain.vectorstores import Chroma

from core.util import download, find_selector

load_dotenv()


@pytest.mark.parametrize(
    "url,selector",
    [
        (
            "https://www.gulte.com/political-news/260908/its-time-to-shift-the-gear-ys-jagan",
            ".entry",
        ),
        (
            "https://www.greatandhra.com/movies/news/senior-actor-couldnt-watch-rrr-pushpa-132442",
            ".great_andhra_main_body_container .page_news .unselectable",
        ),
    ],
)
def test_find_selector(url: str, selector: str):
    assert find_selector(url) == selector


def test_download():
    download(
        "https://www.gulte.com/moviereviews/257842/miss-shetty-mr-polishetty-movie-review",
        ".main-content.single-page-content",
    )
    ...


def test_webbase_loader():
    # Vectorstore
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
    )

    # LLM
    llm = ChatOpenAI(temperature=0)

    # Search
    search = GoogleSearchAPIWrapper()
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore, llm=llm, search=search
    )
    import logging

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
    from langchain.chains import RetrievalQAWithSourcesChain

    user_input = "Who is kilari rajesh who worked for chandra babu naidu?"
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm, retriever=web_research_retriever
    )
    result = qa_chain({"question": user_input})
    result
