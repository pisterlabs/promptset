import openai
import re
import streamlit as st
import pandas as pd


GEN_SQL = """
You will be acting as an AI Snowflake SQL expert named GenSQL.
Your goal is to give correct, executable SQL queries to users.
You will be replying to users who will be confused if you don't respond in the character of GenSQL.
You are given multiple tables with their respective columns.
The user will ask questions; for each question, you should respond and include a SQL query based on the question and the tables provided to you. 

{context}

Here are 6 critical rules for the interaction you must abide:
<rules>
1. You MUST wrap the generated SQL queries within ``` sql code markdown in this format e.g
```sql
(select 1) union (select 2)
```
2. If I don't tell you to find a limited set of results in the sql query or question, you MUST limit the number of responses to 10.
3. Text / string where clauses must be fuzzy match e.g ilike %keyword%
4. Make sure to generate a single Snowflake SQL code snippet, not multiple. 
5. You should only use the table columns given in <columns>, and the table given in <tableName>, you MUST NOT hallucinate about the table names.
6. DO NOT put numerical at the very front of SQL variable.
</rules>

Don't forget to use "ilike %keyword%" for fuzzy match queries (especially for variable_name column)
and wrap the generated sql code with ``` sql code markdown in this format e.g:
```sql
(select 1) union (select 2)
```

For each question from the user, make sure to include a query in your response.

Now to get started, please briefly introduce yourself, show the list of tables available to you , and share the available metrics in 2-3 sentences.
"""

@st.cache_data(show_spinner=False)
def get_cols():

    session = st.experimental_connection("snowpark").session
    table_names = pd.DataFrame(session.sql(f"""
        SELECT DISTINCT TABLE_NAME FROM ML_APP.INFORMATION_SCHEMA.COLUMNS
        Where TABLE_SCHEMA <> 'INFORMATION_SCHEMA'
        ORDER BY TABLE_NAME;
        """,
    ).collect())

    tables = []
    for i in range(len(table_names["TABLE_NAME"])):
        tables.append(table_names["TABLE_NAME"][i]) 

    tables = ["AFD_DATA","CUSTOMER_DETAILS"]

    op = "Tables are given below : \n"
    for j in tables:
        session = st.experimental_connection("snowpark").session
        columns = pd.DataFrame(session.sql(f"""
        SELECT COLUMN_NAME FROM ML_APP.INFORMATION_SCHEMA.COLUMNS
        Where TABLE_NAME ='{j}';
        """,
        ).collect())
    
        cols_metadata = ",".join(
                        [
                f"{columns['COLUMN_NAME'][j]}"
                for j in range(len(columns["COLUMN_NAME"]))
                        ]
                                )
        op = op + f"\n <tableName> {j} <tableName>" + " columns are: <columns>" + cols_metadata + " <columns> \n"
        
    return GEN_SQL.format(context=op)


if __name__ == "__main__":

    st.title("SQL Generator")

    # Initialize the chat messages history
    openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    if "messages" not in st.session_state:
        # system prompt includes table information, rules, and prompts the LLM to produce
        # a welcome message to the user.
        st.session_state.messages = [{"role": "system", "content": get_cols()}]

    # Prompt for user input and save
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})

    # display the existing chat messages
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "results" in message:
                st.dataframe(message["results"])

    # If last message is not from assistant, we need to generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            ):
                response += delta.choices[0].delta.get("content", "")
                resp_container.markdown(response)

            message = {"role": "assistant", "content": response}
            # Parse the response for a SQL query and execute if available
            sql_match = re.search(r"```sql\n(.*)\n```", response, re.DOTALL)
            if sql_match:
                sql = sql_match.group(1)
                session = st.experimental_connection("snowpark").session
                message["results"] = pd.DataFrame(session.sql(sql).collect())
                st.dataframe(message["results"])
                st.session_state["GenSQL_op_df"] = message["results"]
            st.session_state.messages.append(message)