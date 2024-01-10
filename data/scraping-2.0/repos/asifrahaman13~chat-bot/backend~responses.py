import os
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain import (LLMMathChain,SerpAPIWrapper,SQLDatabase,SQLDatabaseChain,)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Use the langchain openai model 
llm = ChatOpenAI(
    temperature=0.7, model="gpt-4", openai_api_key=OPENAI_API_KEY, max_tokens=150
)
# Use the google serp api wrapper
search = SerpAPIWrapper()

# Use langchain math search chain for mathematical calculations. 
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)


async def getResponses(question, query):
    db = SQLDatabase.from_uri(f"sqlite:///userdata.db")
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    
    # We have three tools available. Searh, calcaulator and the data base. The searh is using the serpai for google search. Calculator is to perform the mathematical calcualtions. The data base on the other hand searches through the SQL data base.
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
        Tool(
            name="users",
            func=db_chain.run,
            description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
        ),
    ]
     # This is the memory maangement response. 
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    # Use the conversational buffer memory for the memory management. 
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )
    # res=agent.run(f"First Search through the database and try to find the context to the question. use only the rows with wher the value of username filed is: '{query.username}. If it is present then give asnwer based on that. Else do anything else."+question)
    res = agent.run(
        f"First try to answer your own without the table or database. If you can't find a good answer then Search through the database and try to find the context to the question. use only the rows with wher the value of username filed is: '{query.username}. If that still doesnt work then make a websearch thorugh the serpapi. Do not mention the username in your response. "
        + question
    )
    return res


