from langchain.chat_models import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from tools import execute_sql,get_table_columns,get_table_column_distr

def get_agent_analyst():
    """
    Create and return an analyst agent configured with a predefined prompt and language model.

    This function initializes the necessary components, such as SQL functions, system message, analyst prompt,
    language model (ChatOpenAI), and an agent configured with a pipeline of components.

    Returns:
        dict: Analyst agent configuration.

    Example:
        analyst_agent = get_agent_analyst()
        # Use analyst_agent in your application.
    """


    sql_functions = list(map(format_tool_to_openai_function, [execute_sql, get_table_columns, get_table_column_distr]))


    with open("system_message.txt", "r") as file:
        system_message = file.read()

    analyst_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(temperature=0.1, model = 'gpt-3.5-turbo')\
    .bind(functions = sql_functions)



    analyst_agent = (
    {
        "question": lambda x: x["question"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
    }
    | analyst_prompt
    | llm
    | OpenAIFunctionsAgentOutputParser()
    )

    return analyst_agent

