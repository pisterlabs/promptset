from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import requests

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from links_suggestor import LinksSuggestor
from utils import get_links
from api_executor import APIExecutor
import json

import click
from utils import (
    API_PROMPT_TEMPLATE,
    CODE_PROMPT_TEMPLATE,
    LINK_PROMPT_TEMPLATE,
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


class APIReader:
    """
    Stores API Documentation
    """

    def __init__(self):
        """
        Params:
            - src_url: url of api documentation main page
        """
        self.db = Chroma(
            collection_name=f"api_documentation",
            embedding_function=embedding_function,
        )
        self.links = set([])
        self.link_suggestor = LinksSuggestor(self.db, self.links)

        # ingest api documentation
        # self.load_documentation(src_url, max_depth=0)

    def get_db(self):
        return self.db

    def suggest_links(self, query):
        res = self.link_suggestor.suggest_links(query)

        return res

    def load_documentation(self, root: str, max_depth=1):
        """ "
        Performs BFS to load documentation
        """

        # (url, level)
        queue = [(root, 0)]
        visited = set([root])

        while queue:
            url, level = queue.pop(0)

            self.ingest_documentation(url)

            # get links from url
            neighbor_links = get_links(url)

            for link in neighbor_links:
                if link not in visited and level + 1 <= max_depth:
                    visited.add(link)
                    queue.append((link, level + 1))

    def ingest_documentation(self, url: str):
        res = requests.get(url)
        content_type = res.headers["content-type"]

        # we only want html
        if not content_type.startswith("text/html"):
            return

        print("INGESTING")

        urls = [url]
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True,
        )

        doc_texts = [doc.page_content for doc in docs_transformed]
        metadatas = [{"source": url} for doc in docs_transformed]
        texts = text_splitter.create_documents(texts=doc_texts, metadatas=metadatas)

        self.db.add_documents(texts)
        self.links.add(url)


class Verifier:
    """
    Checks if documents contain enough information to answer a query
    """

    """
    ideas on how to verify (joso):
    1. attempt to answer the query with the existing DS. if it works, it's good. 
       otherwise, feed the error back into the system and retrain.
    2. multi-llm verification: q: "can i answer the document given this info?" a:
    """

    def __init__(self, db: Chroma, llm=None):
        self.db = db
        self.api_prompt = API_PROMPT_TEMPLATE
        self.code_prompt = CODE_PROMPT_TEMPLATE
        self.step_feedback_prompt = LINK_PROMPT_TEMPLATE
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(model_name="gpt-4")

        self.api_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            retriever=db.as_retriever(),
            chain_type_kwargs={"prompt": self.api_prompt},
        )

        self.code_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            retriever=db.as_retriever(),
            chain_type_kwargs={"prompt": self.code_prompt},
        )

        self.step_feedback_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            retriever=db.as_retriever(),
            chain_type_kwargs={"prompt": self.step_feedback_prompt},
        )

    def verify(self, prompt: str):
        """
        Params:
            - prompt: prompt to verify
        """
        if prompt.startswith("API"):
            print(f"using api prompt")
            result = self.api_chain({"question": prompt})
        else:
            print(f"using code prompt")
            result = self.code_chain({"question": prompt})
        answer = result["answer"]
        sources = result["sources"]
        return answer, sources  # if cannot answer, then says "I don't know"

    # def step_feedback(self, prompt: str, answer: str, feedback: str):
    #     """
    #     Params:
    #         - prompt: prompt to verify
    #         - feedback: feedback to train on
    #     """
    #     answer, sources = self.verify(prompt)
    #     print(answer)
    #     if answer == "CANNOT DO":  # prompt for a series of links
    #         result = self.step_feedback_chain({"question": prompt})
    #         answer, sources = result["answer"], result["sources"]
    #         print(answer)
    #     return answer, sources


@click.command()
@click.option("--url", default="https://jsonplaceholder.typicode.com")
def main(url):
    reader = APIReader(url)
    verifier = Verifier(reader.get_db())

    while True:
        prompt = click.prompt("What do you want to do?")
        for _ in range(3):
            answer, sources = verifier.verify(prompt)
            print(answer)
            answer = json.loads(answer)
            if answer["failure_and_suggestions"] == "CANNOT DO":
                link = reader.suggest_links(prompt)
                reader.ingest_documentation(link)
            else:
                break
        print(answer)
        print("---")
        print(sources)

        api_executor = APIExecutor()
        api_params = json.loads(answer)

        api_result = api_executor.execute(api_params)

        print(api_result)


if __name__ == "__main__":
    main()
