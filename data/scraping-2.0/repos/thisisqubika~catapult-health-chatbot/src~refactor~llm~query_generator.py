from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import streamlit as st
from snowflake.snowpark import Session
from sqlalchemy.dialects import registry
from langchain_experimental.sql import SQLDatabaseChain
from llm.streamming_handler import StreamHandler
from config import OPENAI_API_KEY,db

LLM = ChatOpenAI(
    temperature=0.0,
    model="gpt-4-1106-preview",
    streaming=True,
    openai_api_key=OPENAI_API_KEY
)


QUALIFIED_TABLE_NAME = "CATAPULT_HEALTH_DB.POC_CATAPULT_HEALTH.HEALTHRECORDDATA"

TABLE_DESCRIPTION = """
This table is an electronic health record (EHR) system or a patient health database which has clinical or healthcare data from patients.
"""

METADATA_QUERY = "SELECT VARIABLE_NAME, DEFINITION FROM CATAPULT_HEALTH_DB.POC_CATAPULT_HEALTH.HEALTHRECORDDATA_ATTRIBUTES;"


SYSTEM_TEMPLATE = """
Your goal is to give correct, executable Snowflake sql query to users.
You will be acting as an AI Snowflake SQL Expert named Catapult Health Bot.
You will be replying to users who will be confused if you don't respond in the character of Catapult Health Bot.
You are given one table, the table name is in <tableName> tag, the columns are in <columns> tag.
The user will ask questions, for each question you should return only the snowflake sql query based on the question and the table, delimited by triple backticks.
Remember: The "FILTER" function does not exist in snowflake sql. Just create sql queries that work on Snowflake, you must use existing snowflake SQL syntax

{context}

Here are 9 critical rules for the interaction you must abide:
<rules>
1. The code MUST BE SNOWFLAKE SQL, it MUST NOT BE other types of SQL, such as bigquerySQL, or POSTGRESQL.
2. You MUST MUST wrap the generated snowflake sql code within ``` sql code markdown in this format e.g
```sql
(select 1) union (select 2)
```
3. If you get more than one question in a request, you should provide different sql codes for each question and each sql code should be provided within it's own ``` sql code markdown  
4. If I don't tell you to find a limited set of results in the sql query or question, you MUST NOT limit the number of responses.
5. Text / string where clauses must be fuzzy match e.g ilike %keyword%
6. Make sure to generate a single snowflake sql code, not multiple. 
7. You should only use the table columns given in <columns>, and the table given in <tableName>, you MUST NOT hallucinate about the table names
8. DO NOT put numerical at the very front of sql variable.
9. If you get multiple questions requiering more than 1 sql code response, answer back each sql code between it's own markdown:
For example:

user asks: How many patients for Company One have been invited to schedule their NP consult but haven’t done so? 
I also need a list of patients for Company One who missed their NP consults

your answer:
```sql
SELECT COUNT(*) AS Patients_Invited_But_Not_Scheduled
FROM CATAPULT_HEALTH_DB.POC_CATAPULT_HEALTH.HEALTHRECORDDATA
WHERE COMPANYNAME ILIKE '%Company One%'
AND INVITATIONSENT = TRUE
AND CONSULTCOMPLETED = FALSE;
```
```sql
SELECT HEALTHEVENTID, COMPANYNAME, INVITATIONSENT, CONSULTCOMPLETED
FROM CATAPULT_HEALTH_DB.POC_CATAPULT_HEALTH.HEALTHRECORDDATA
WHERE COMPANYNAME ILIKE '%Company One%'
AND INVITATIONSENT = TRUE
AND CONSULTCOMPLETED = FALSE;
```

</rules>

Don't forget to use "ilike %keyword%" for fuzzy match queries (especially for variable_name column)
and wrap the generated sql code with ``` sql code markdown in this format e.g:
```sql
(select 1) union (select 2)
```
For each question from the user, make sure to include a query in your response.

DO NOT RETURN ANYTHING ELSE. JUST THE SNOWFLAKE SQL QUERY.

User input:
{user_input}

Your generated snowflake query:
"""


@st.cache_data(show_spinner=False)
def get_table_context(_snowflake_session: Session, table_name: str, table_description: str, metadata_query: str = None ):
    table = table_name.split(".")
    # Execute the query
    query = f"""
        SELECT COLUMN_NAME, DATA_TYPE FROM {table[0].upper()}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{table[1].upper()}' AND TABLE_NAME = '{table[2].upper()}'
    """
    result = _snowflake_session.sql(query).collect()

    # Process the result
    columns = "\n".join(
        [
            f"- **{row['COLUMN_NAME']}**: {row['DATA_TYPE']}"
            for row in result
        ]
    )

    context = f"""
    Here is the table name <tableName> {'.'.join(table)} </tableName>
    <tableDescription>{table_description}</tableDescription>
    Here are the columns of the {'.'.join(table)}
    <columns>\n\n{columns}\n\n</columns>
    """

    if metadata_query:
        metadata_result = _snowflake_session.sql(metadata_query).collect()
        metadata = "\n".join(
            [
                f"- **{row['VARIABLE_NAME']}**: {row['DEFINITION']}"
                for row in metadata_result
            ]
        )
        context += f"\n\nAvailable variables by VARIABLE_NAME:\n\n{metadata}"

    return context




class QueryGeneratorLLM:
    def __init__(self, _snowflake_session:Session, llm: BaseLanguageModel = LLM,system_template: str = SYSTEM_TEMPLATE):
        self.llm = llm
        self.template = system_template
        self.snowflake_session = _snowflake_session
        self.stream_handler = StreamHandler(st.empty())

    def _get_connection(self):
        # create connection:
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=db,
            verbose=True,
            return_direct=True)
        
        return db_chain.run
    
    def _build_prompt(self, user_input: str) -> PromptTemplate:
        table_context = get_table_context(
            _snowflake_session=self.snowflake_session,
            table_name=QUALIFIED_TABLE_NAME,
            table_description=TABLE_DESCRIPTION,
            metadata_query=METADATA_QUERY)

        # Now format the template with both context and user_input
        formatted_template = self.template.format(context=table_context, user_input=user_input)

        return str(PromptTemplate.from_template(
            template=formatted_template,
            variables=["user_input"],
        ))

    def generate_sql_query(self, user_input: str) -> str:
        chain_instance = self._build_chain(user_input)  # Pass user input here
        # return chain_instance.predict(text=user_input)
        response_stream = chain_instance.predict(text=user_input)
        response = ""
        for chunk in response_stream:
            response += chunk
        return response


    def _build_chain(self, user_input: str) -> LLMChain:
        return LLMChain(
            llm=self.llm,
            prompt=self._build_prompt(user_input),  # Pass user input here
            verbose=True
            )
    
    def generate_response(self, user_input: str)-> str:
        prompt = self._build_prompt(user_input)
        response_stream = self.llm.stream(prompt)

        # Clear the existing text in the stream handler
        self.stream_handler.text = ""

        # Stream the response and let the handler update the text
        for chunk in response_stream:
            self.stream_handler.on_llm_new_token(chunk)

        # The final response should now be in self.stream_handler.text
        # Remove any unwanted parts from the response if necessary
        response = self.stream_handler.text.replace("content=", "").strip()

        return response


    