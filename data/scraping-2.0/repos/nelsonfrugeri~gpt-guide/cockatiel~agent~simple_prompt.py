import os

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSerperAPIWrapper

def main():
    user = os.getenv("USER")

    tools = [
        Tool(
            name="Google Serper API",
            func=GoogleSerperAPIWrapper().run,
            description="useful for when you need to ask with search",
        )
    ]

    agent = ZeroShotAgent(
        llm_chain=LLMChain(
            llm=OpenAI(temperature=os.getenv("OPENAI_PARAM_TEMPERATURE")), 
            prompt=ZeroShotAgent.create_prompt(
                tools,
                prefix="""Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:""",
                suffix="""Begin!"
                    {chat_history}
                    Question: {input}
                    {agent_scratchpad}""",
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )
        ), 
        tools=tools, verbose=os.getenv("GOOGLE_SERPER_API_PARAM_VERBOSE")
    )

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=os.getenv("GOOGLE_SERPER_API_PARAM_VERBOSE"), 
        memory=ConversationBufferMemory(memory_key="chat_history")
    )

    print('Chat: Hello, what can I do for you?')

    while True:
        user_input = input(f'{user}: ')

        if (user_input.lower() == 'exit' or 
            user_input.lower() == 'clear'):

            print('Leaving chat... bye')
            break

        print(agent_chain.run(input=user_input))

if __name__ == '__main__':
    main()