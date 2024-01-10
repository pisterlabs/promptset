from langchain.chat_models import ChatOpenAI
import wikipedia

class GeneralKnowledgeService:
    def __init__(self, chat: ChatOpenAI):
        self.chat = chat

    def search_page(self, search: str):
        return wikipedia.search(search, True)

    def get_wikipedia_page(self, search: str) -> str:
        return wikipedia.page(search).content

    def get_wikipedia_summary(self, search: str) -> str:
        return wikipedia.summary(search)
