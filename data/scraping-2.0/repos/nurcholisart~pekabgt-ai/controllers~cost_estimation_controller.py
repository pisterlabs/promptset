import tiktoken
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

ENCODING = "gpt-4"
COST_PER_1000_TOKEN = 0.0004


class CostEstimationController:
    def call(self, contents: list[str]):
        plain_contents = self._convert_html_to_plain(contents)
        documents = self._generate_langchain_documents(plain_contents)
        documents = self._split_character_text(documents)

        enc = tiktoken.encoding_for_model(ENCODING)

        word_count = sum(len(doc.page_content.split()) for doc in documents)
        token_count = sum(len(enc.encode(doc.page_content)) for doc in documents)

        cost = token_count * COST_PER_1000_TOKEN / 1000
        cost = "{:.7f}".format(cost)

        return {"word_count": word_count, "token_count": token_count, "cost": cost}

    def _generate_langchain_documents(
        self, plain_contents: list[str]
    ) -> list[Document]:
        documents = []

        for content in plain_contents:
            document = Document(page_content=content, metadata={})
            documents.append(document)

        return documents

    def _convert_html_to_plain(self, contents: list[str]) -> list[str]:
        plain_contents = []

        for content in contents:
            soup = BeautifulSoup(content, "html.parser")
            plain_content = soup.get_text()

            plain_contents.append(plain_content)

        return plain_contents

    def _get_cost_by_encoding(self, encoding: str):
        encodings = {"gpt-4": 0.0004}

        cost_per_1000_token = encodings[encoding]
        if cost_per_1000_token is None:
            cost_per_1000_token = 0.0004

        return cost_per_1000_token

    def _split_character_text(self, documents: list[Document]) -> list[Document]:
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
        )

        splitted_documents = text_splitter.split_documents(documents)
        return splitted_documents
