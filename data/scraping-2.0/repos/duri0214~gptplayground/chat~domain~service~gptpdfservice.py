from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain

from chat.domain.valueobject.dataloader import Dataloader


class GptPdfService:
    def __init__(self, dataloader: Dataloader, n_results: int = 3):
        self.dataloader = dataloader

        self.n_results = n_results
        self.system_template = """
            以下の資料の注意点を念頭に置いて回答してください
            ・ユーザの質問に対して、できる限り根拠を示してください
            ・箇条書きで簡潔に回答してください。
            ---下記は資料の内容です---
            {summaries}

            Answer in Japanese:
        """
        messages = [
            SystemMessagePromptTemplate.from_template(self.system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        self.prompt_template = ChatPromptTemplate.from_messages(messages)

    @staticmethod
    def _create_vectorstore(dataloader: Dataloader) -> Chroma:
        """
        Note: OpenAIEmbeddings runs on "text-embedding-ada-002"
        """
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            dataloader.data,
            embedding=embeddings,
            persist_directory='.'
        )
        vectorstore.persist()

        return vectorstore

    def gpt_answer(self, user_text: str, chat_history: List[str]) -> dict:
        """
        Note: ChatOpenAI runs on 'gpt-3.5-turbo'
        """
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
        texts = [x.page_content for x in self.dataloader.data]
        metadatas = [x.metadata for x in self.dataloader.data]
        docsearch = Chroma.from_texts(texts, embeddings, metadatas)
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                            chain_type="stuff",
                                                            reduce_k_below_max_tokens=True,
                                                            return_source_documents=True,
                                                            retriever=docsearch.as_retriever(),
                                                            chain_type_kwargs={"prompt": self.prompt_template})

        return chain({"question": user_text})
