'''
import streamlit as st
import os
import openai

def codex():
    st.write("Enter your query below and press the run button")
    codex_context = "--Snowflake SQL Query" + "\n" + "--Add quotation marks around table names and column names"
    table_name = "--" + st.session_state['table']
    query = "--" + st.text_area("Query") + "\n" + "\n" 
    codex_request = codex_context + "\n" + table_name + "\n" + query



    openai.api_key = st.text_input('API Token', 'Enter OpenAI API Token', type="password")

    if st.button('Generate Code'):
        response = openai.Completion.create(
          engine="code-davinci-002",
          prompt=codex_request,
          temperature=0,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0.5,
          presence_penalty=0.5
        )

    st.code(response.choices[0].text, language="sql")

    st.session_state['user']=st.text_input('Snowflake Username', 'Enter Snowflake Username')
    st.session_state['password']=st.text_input('Snowflake Password', 'Enter Snowflake Password', type="password")
    st.session_state['account']=st.text_input('Snowflake Account', 'Enter Snowflake Account')
    st.session_state['warehouse']=st.text_input('Snowflake Warehouse', 'Enter Snowflake Warehouse')
    st.session_state['database']=st.text_input('Snowflake Database', 'Enter Snowflake Database')
    st.session_state['schema']=st.text_input('Snowflake Schema', 'Enter Snowflake Schema')


    if st.button('Run Query'):
        conn = snowflake.connector.connect(
        user = st.session_state['user'],
        password = st.session_state['password'],
        account = st.session_state['account'],
        warehouse = st.session_state['warehouse'],
        database = st.session_state['database'],
        schema = st.session_state['schema']
        )
        cur = conn.cursor()
        cur.execute(response.choices[0].text)


if __name__ == '__main__':
    codex()

'''