from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import streamlit as st
from snowflake.snowpark import Session
from langchain_experimental.sql import SQLDatabaseChain
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
Your goal is to create charts or graphs for users request with python code based on Snowflake SQL Queries you will recieve, 
you MUST use the Plotly python library for this tasks.
You will recieve a list of ROW objects. The Row object is a built-in data structure in many data processing libraries, including pandas in Python.
You need to convert these list of ROW objects into charts or graphs with Plotly and pandas libraries from python.
The user will provide you instructions for the creation of the charts and also you will recieve the data to create those charts as a list of ROW objects, for each question you should return only the charts or graphs based on the list of ROW ojects delimited by triple backticks.
You should be able to provide different types of charts, depending of the user's request:
pie charts, histograms, bar charts, line charts, etc.

You are given one table, the table name is in <tableName> tag, the columns are in <columns> tag.

{context}

Here are 4 critical rules for the interaction you must abide:
<rules>
1. You MUST MUST wrap the generated python code within ``` python code markdown in this format e.g
```python
import pandas as pd
df = pd.DataFrame([(2878, datetime.date(2023, 2, 22)), (2909, datetime.date(2023, 2, 23)), (2977, datetime.date(2023, 2, 24)), (4184, datetime.date(2023, 2, 21)), (25123, datetime.date(2023, 2, 20)), (31730, datetime.date(2023, 2, 19)), (24394, datetime.date(2023, 2, 18))], columns=['cantidad_lineas_diferentes', 'fecha'])
st.bar_chart(df, x='fecha', y='cantidad_lineas_diferentes')
```
2. Use python's Plotly library to make the charts, whenever the input has one of the following sentences: 
show a bar chart..., show a pie chart..., show a line chart... plot a bar chart..., plot a pie chart..., plot a line chart... In every response, you will have to show charts. 
3. Don't explain the process and the python code you've used,just show the Chart required for the user. You MUST NOT be verbose on the explanation, just show the results.
4. to solve the task, import only the necesary libraries... they will usually be Pandas, Plotly, Datetime and a few more.. only import the ones that you need.
</rules>


For each User request containing the chunk of data recieved in a list of ROW objects , make sure to include a chart in your response.

DO NOT RETURN ANYTHING ELSE. JUST THE CHARTS.

User input:
{user_input}

these is the list of ROW objects:
{sql_query}

Your generated chart:
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




class ChartGeneratorLLMFromSQL:
    def __init__(self, _snowflake_session:Session, llm: BaseLanguageModel = LLM,system_template: str = SYSTEM_TEMPLATE):
        self.llm = llm
        self.template = system_template
        self.snowflake_session = _snowflake_session

    def _get_connection(self):
        # create connection:
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=db,
            verbose=True,
            return_direct=True)
        
        return db_chain.run
    
    def _build_prompt(self, user_input: str, sql_query: str) -> PromptTemplate:
        table_context = get_table_context(
            _snowflake_session=self.snowflake_session,
            table_name=QUALIFIED_TABLE_NAME,
            table_description=TABLE_DESCRIPTION,
            metadata_query=METADATA_QUERY)

        # Now format the template with both context and user_input
        formatted_template = self.template.format(context=table_context, user_input=user_input, sql_query=sql_query)

        return PromptTemplate.from_template(
            template=formatted_template,
            variables=["user_input","sql_query"],
        )

    def generate_chart(self, user_input: str, sql_query: str) -> str:
        chain_instance = self._build_chain(user_input,sql_query)  # Pass user input here
        # return chain_instance.predict(text=sql_query)
        response_stream = chain_instance.predict(user_input=user_input, sql_query=sql_query)
        response = ""
        for chunk in response_stream:
            response += chunk
        return response


    def _build_chain(self, user_input: str ,sql_query: str) -> LLMChain:
        return LLMChain(
            llm=self.llm,
            prompt=self._build_prompt(user_input, sql_query),  # Pass user input here
            verbose=True
            )
