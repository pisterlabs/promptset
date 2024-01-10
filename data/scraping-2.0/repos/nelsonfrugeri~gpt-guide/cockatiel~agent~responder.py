import os

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain import LLMMathChain

def main():
    user = os.getenv("USER")
    
    tools = [
        Tool(
            name="Search",
            func=DuckDuckGoSearchAPIWrapper().run,
            description="useful for when you need to ask with search",
        ),
        Tool(
            name="Wikipedia",
            func=WikipediaAPIWrapper().run,
            description="useful when you need an answer about general knowledge"
        ),
        Tool(
            name="Calculator",
            func=LLMMathChain.from_llm(llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo'), verbose=False).run,
            description="useful for when you need to answer questions about math"
        )
    ]

    agent = initialize_agent(
        tools, 
        ChatOpenAI(temperature=0, model='gpt-3.5-turbo'), 
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        agent_kwargs={
            'prefix': """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:""", 
            'suffix': """Begin!"
                {chat_history}
                Question: {input}
                {agent_scratchpad}"""
        },
        verbose=True
    )

    while True:
        user_input = input(f'{user}: ')

        if (user_input.lower() == 'exit' or 
            user_input.lower() == 'clear'):

            print('Leaving chat... bye')
            break

        print(agent.run(user_input))

if __name__ == '__main__':
    main()