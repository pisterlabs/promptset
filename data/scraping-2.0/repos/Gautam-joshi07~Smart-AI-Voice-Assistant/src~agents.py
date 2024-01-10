from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
AgentType,load_tools,initialize_agent
)
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StdOutCallbackHandler

class ConversationalAgent:

    def __init__(self) -> None:
        self.llm = ChatOpenAI()
        self.chain = ConversationChain(llm=self.llm)


    def run(self, text):
        if  len(text) > 14000:
            return self.chain.run(text)
        return self.chain.run(text)




class SmartChatAgent:
    def __init__(self) -> None:

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.llm = ChatOpenAI(temperature=0,max_tokens=14000,model_name="gpt-3.5-turbo-16k")
        self.tools = load_tools(['google-search'])

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,

        )

    def run(self, text):
        handler = StdOutCallbackHandler()
        return self.agent.run(text,callbacks=[handler])