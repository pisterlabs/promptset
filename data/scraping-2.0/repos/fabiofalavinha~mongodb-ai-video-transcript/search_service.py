from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

from openai_service import OpenAIService

SEARCH_INDEX_NAME = ""
SEARCH_TEXT_KEY = ""
SEARCH_EMBEDDING_KEY = ""


class MongoDBSearchService:
    def __init__(self, collectionName: str, openAiService: OpenAIService) -> None:
        self.__langChainEmbeddings = OpenAIEmbeddings(
            model=OpenAIService.SEARCH_ENGINE,
            chunk_size=1,
            deployment=OpenAIService.SEARCH_ENGINE,
            openai_api_key=openAiService.getOpenAIKey(),
            openai_api_base=openAiService.getOpenAIBase(),
            openai_api_type=openAiService.getOpenAIType(),
            openai_api_version=openAiService.getOpenAIVersion()
        )
        self.__vectorStore = MongoDBAtlasVectorSearch(
            collectionName,
            self.__langChainEmbeddings,
            index_name=SEARCH_INDEX_NAME,
            text_key=SEARCH_TEXT_KEY,
            embedding_key=SEARCH_EMBEDDING_KEY
        )
        self.__template = \
            "Use the following pieces of context, where each piece is a text representation of a similarity search, " \
            "to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to " \
            "make up an answer. Don't include the prefix 'Answer:' in the answer. Don't return any 'Context:' pieces in the answer. " \
            "Don't include any questions in the answer. Just output a very straightforward and clean answer.\n{context}\n\n{question}"
        self.__promptTemplate = PromptTemplate(
            template=self.__template,
            input_variables=["context", "question"]
        )
        self.__chain = RetrievalQA.from_llm(
            llm=ChatOpenAI(
                temperature=0,
                model_name=OpenAIService.CHAT_MODEL,
                openai_api_key=openAiService.getOpenAIKey()
            ),
            retriever=self.__vectorStore.as_retriever(),
            prompt=self.__promptTemplate
        )

    def createEmbedding(self, text: str) -> [float]:
        try:
            return self.__langChainEmbeddings.embed_query(text)
        except Exception as ex:
            raise Exception(
                f"Error creating embedding by text [{text}] using engine [{OpenAIService.SEARCH_ENGINE}]: {ex}")

    def searchBy(self, prompt: str) -> str:
        if prompt is None or len(prompt) == 0:
            return ""
        completion = self.__chain.run(prompt)
        completionLines = completion.split("\n")
        return '\n'.join(completionLines)
