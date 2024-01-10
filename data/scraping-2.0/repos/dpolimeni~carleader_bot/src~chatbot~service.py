from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, Tool, initialize_agent
from src.chatbot.service_abc import BaseQA
from src.schemas import OpenaiConfig
from typing import List, Any

## TODO check if we need a retriever


class QaService(BaseQA):
    def __init__(
        self, openai_config: OpenaiConfig, local_models: List[Any] = None
    ) -> None:
        """Init is done with the openai key and local models (whatever they are)"""

        self.chat_llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_config.openai_key,
            model=openai_config.chat_model_version,
            request_timeout=15,
        )

        self.emb_llm = OpenAIEmbeddings(
            openai_api_key=openai_config.openai_key, client=None
        )

    async def basic_answer(self, query: str, context: str) -> str:
        messages = [SystemMessage(content=context), HumanMessage(content=query)]

        response = self.chat_llm(messages)

        return response.content

    def init_agent(self, tools: List[Tool] = None):
        # agent_executor = create_conversational_retrieval_agent(
        #    tools=tools,
        #    llm=self.chat_llm,
        #    verbose=True,
        #    SystemMessage="""Sei l'assistente di un concessionario ed il tuo compito proporre ai clienti le macchine
        # più in linea con le loro richieste. Sei dettagliato nella descrizione delle auto da proporre.
        # Quando proponi una macchina al cliente descrivigli alcune caratteristiche ed allega sempre il link dell'auto.
        # Se il cliente è troppo generico nella richiesta chiedigli di fornire più informazioni sul genere di auto che gli interessa.
        # """,
        #         )
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        agent_executor = initialize_agent(
            tools,
            self.chat_llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
        )
        return agent_executor
