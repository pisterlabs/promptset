"""Chain Service"""

from langchain.agents import AgentType, initialize_agent
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

from src.utils import get_chat_history, get_system_template


class ChainService:
    """Chain Service"""
    def __init__(self, model):
        self.model = model

    def condense_question(self):
        """Condense a question into a single sentence."""
        return LLMChain(
			llm=self.model,
			prompt=CONDENSE_QUESTION_PROMPT,
		)

    def collect_docs(self, system_message):
        """Collect documents from the vectorstore."""
        return load_qa_chain(
			self.model,
			chain_type='stuff',
			prompt=get_system_template(system_message)
		)

    def conversation_retrieval(
        self,
		vectorstore,
		system_message
    ):
        """Retrieve a conversation."""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain(
			question_generator=self.condense_question(),
			retriever=vectorstore.as_retriever(),
			memory=memory,
			combine_docs_chain=self.collect_docs(system_message),
			get_chat_history=get_chat_history,
		)

    def agent_search(self, tools, chat_history):
        """Agent search."""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        if len(chat_history) > 0:
            for message in chat_history:
                if message[0] and message[1]:
                    memory.chat_memory.add_user_message(message[0])
                    memory.chat_memory.add_ai_message(message[1])
                else:
                    memory.chat_memory.add_user_message(message[0])   
        return initialize_agent(
            tools,
            self.model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION or AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            get_chat_history=get_chat_history
        )
  