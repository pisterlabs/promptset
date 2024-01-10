from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI

from dotsavvy.services.vectorstore import create_pinecone_vectorstore
from dotsavvy.utils.env_variables import get_env_variable


class ConversationAgent:
    def __init__(self) -> None:
        openai_api_key: str = get_env_variable("OPENAI_API_KEY")
        llm_name: str = get_env_variable("DOTSAVVY_LLM_NAME")
        self.chat_model = ChatOpenAI(openai_api_key=openai_api_key, model_name=llm_name)
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=5, return_messages=True
        )
        vectorstore = create_pinecone_vectorstore()
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
        )

    def run(self, query: str) -> str:
        return self.retrieval_qa.run(query=query)
