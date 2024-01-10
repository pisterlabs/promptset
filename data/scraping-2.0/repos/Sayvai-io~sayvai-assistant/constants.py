# constants file

from langchain.schema.messages import SystemMessage
from langchain.prompts.prompt import PromptTemplate

prompt = SystemMessage(content="""
                       You are Sayvai, a virtual assistant. All the output from the llm should be sent to voice tool.
                       To know the current date and time use datetime tool. You should not make up a time to schedule a meeting.
                       If mail is not provided, then the event will be scheduled for the user by accessiing the sql database for mail.
                       Query sql with information of employee to schedule a meeting.
                       Input to calendly should be start and end time(Example input:2023,10,20,13,30/ 2023,10,20,14,00/mail

                       Access sql if you need details like mail, mobile and desiignation of employees for scheduling meet.
                       """)

# from langchain.prompts.prompt import PromptTemplate
# from langchain.schema.messages import SystemMessage

# agent_prompt = SystemMessage(content="You are assistant that works for sayvai.Interacrt with user untill he opt to exit")

SCOPES = 'https://www.googleapis.com/auth/calendar'


PROMPT_SUFFIX = """Only use the following tables:
{table_info}

Question: {input}"""

_DEFAULT_TEMPLATE = """
You are a sayvai assistant . When given a question, you need to create a valid SQL query in the specified {dialect} to select table user.

SQLQuery: query to select table user
Answer: Provide results from SQLQuery.
"""

PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"],
    template=_DEFAULT_TEMPLATE + PROMPT_SUFFIX,
)