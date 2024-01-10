from tools.database import create_connection, execute_read_query

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def lookup(query: str) -> str:
    print("The query string is: " + query + "\n")

    db_name = "analytics"
    db_user = "citus"
    db_password = "citus"
    db_host = "localhost" 
    db_port = "5433" 

    # Create the connection
    connection = create_connection(db_name, db_user, db_password, db_host, db_port)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """given the {query} I want you to get it me .
    make the {query} on database with {connection} and return results ."""

    tools_for_agent = [
        Tool(
            name="Query the database with {query}",
            func=execute_read_query(connection=connection, query=query),
            description="useful for when you need to query to the database with given sql.",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt_template = PromptTemplate(template=template, input_variables=["query"])

    the_result = agent.run(prompt_template.format_prompt(query=query))
    connection.close()
    return the_result
