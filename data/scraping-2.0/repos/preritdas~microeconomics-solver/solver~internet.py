"""Internet tools, like searching Google and reading websites."""
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from bs4 import BeautifulSoup
import requests
import json

from abc import ABC, abstractmethod
from keys import KEYS


llm = ChatOpenAI(model_name="gpt-4", openai_api_key=KEYS.OpenAI.api_key, temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=KEYS.OpenAI.api_key)
N_DOCS = 10  # 10 for gpt-4, 5 for 3.5
splitter = TokenTextSplitter(
    encoding_name="cl100k_base", chunk_size=300, chunk_overlap=50
)


REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:12.0) " "Gecko/20100101 Firefox/12.0"
    ),
    "Accept-Language": "en-US",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "text/html",
    "Referer": "https://www.google.com",
}


class BaseAnswerer(ABC):
    """Abstract base class for answerers."""

    def __init__(self, source: str):
        self.source = source

    @abstractmethod
    def convert(self) -> str:
        """
        Converts `self.source` into a string. If the input is text already, return input.
        If it's a website, scrape the website and return the text. Etc. The input should
        be providable in string form so a GPT agent can use it.

        Ex. text can be passed as a string, website can be passed as a URL, etc.
        """

    def answer(self, query: str, n_docs: int = N_DOCS) -> str:
        """
        First converts the initial source, then queries it. The query must be a string,
        and the answer will be a string. This does not work with the string-in-string-out
        nature of an LLM agent, so it is not exposed to the user.
        """
        text = self.convert()
        docs = splitter.create_documents([text])
        vectorstore = FAISS.from_documents(docs, embeddings)

        _find_similar = lambda k: vectorstore.similarity_search(query, k=k)
        similar_docs = _find_similar(n_docs)

        # Adjust the instructions based on the source
        PREFIX = (
            f"You are a {type(self).__name__}. Your context is snippets from the "
            f"transcription of your source as a {type(self).__name__}. "
        )
        qa_chain = load_qa_chain(llm)
        qa_chain.llm_chain.prompt.messages[0].prompt.template = (
            PREFIX + qa_chain.llm_chain.prompt.messages[0].prompt.template
        )

        return qa_chain.run(input_documents=similar_docs, question=query)

    @classmethod
    def answer_json_string(cls, agent_input: str) -> str:
        """
        Parses the agent input and returns the answer. This is the function that the
        agent will call. The agent input must be a string, and the answer will be a string.
        Parses with JSON. Agent must provide a JSON string with two keys: "source" and "query".

        Eventually, add some JSON parsing to allow for slightly-off inputs.
        """
        dic = json.loads(agent_input)

        # Return the answer or an error
        try:
            return cls(dic["source"]).answer(dic["query"])
        except Exception:
            return "Unfortunately cannot answer questions using that particular source."


class WebsiteAnswerer(BaseAnswerer):
    """Answerer for websites."""

    def convert(self) -> str:
        """Convert website to text."""
        response = requests.get(self.source, headers=REQUEST_HEADERS)
        soup = BeautifulSoup(response.content, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        return " ".join(string for string in soup.stripped_strings)
