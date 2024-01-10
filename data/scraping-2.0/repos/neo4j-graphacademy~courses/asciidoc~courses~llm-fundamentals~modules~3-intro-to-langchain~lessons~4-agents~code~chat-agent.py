from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool

llm = ChatOpenAI(
    openai_api_key="sk-..."
)

prompt = PromptTemplate(
    template="""
    You are a movie expert. You find movies from a genre or plot. 

    ChatHistory:{chat_history} 
    Question:{input}
    """, 
    input_variables=["chat_history", "input"]
    )

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

tools = [
    Tool.from_function(
        name="ChatOpenAI",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=chat_chain.run,
        return_direct=True
    )
]

agent = initialize_agent(
    tools, llm, memory=memory, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
)

while True:
    q = input(">")
    print(agent.run(q))