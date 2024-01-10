"""
Adapted from Frosty
https://quickstarts.snowflake.com/guide/frosty_llm_chatbot_on_streamlit_snowflake
"""

import openai
import re
import streamlit as st
from prompts import get_system_prompt

openai.api_type = "azure"
openai.api_version = "2023-08-01-preview"
openai.api_base = st.secrets["azureopenai"]["AZURE_OPENAI_ENDPOINT"]
openai.api_key = st.secrets["azureopenai"]["AZURE_OPENAI_KEY"]
ENGINE = st.secrets["azureopenai"]["AZURE_OPENAI_DEPLOYMENT_NAME"]


def clear_cache():
    '''
    Clear cache when a new table is selected
    '''
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)


def app():
    '''
    The app with the required UI elements and logic
    '''
    st.set_page_config(
        page_title="Opti-Q",
        page_icon=":rocket:",)
    st.markdown("<h1 style='text-align: center; color: blue;'>Opti-Q</h1>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Your natural language query bot for your ODP Snowflake database</h1>", unsafe_allow_html=True)

    # Example of fully qualified table names
    table_names = ["PROD_1_WEB.PUBLIC.PRODUCTS",
                "PROD_2_WEB.PUBLIC.PRODUCTS", "PROD_3_WEB.PUBLIC.PRODUCTS"]

    # Dropdown to select a table
    selected_table = st.selectbox("Select a table",
                               table_names,
                               on_change=clear_cache,
                               index=None,
                               placeholder="Choose an option")

    st.session_state['selected_table'] = selected_table

    if st.session_state.selected_table:
        # Display the selected table
        st.write(f"You selected selected_table {st.session_state.selected_table}")
        if "messages" not in st.session_state:
            # system prompt includes table information, rules, and prompts the LLM to produce
            # a welcome message to the user.
            st.session_state.messages = [
                {"role": "system", "content": get_system_prompt(st.session_state.selected_table)}]

        # Prompt for user input and save
        if prompt := st.chat_input():
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

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
                    engine=ENGINE,
                    model="gpt-3.5-turbo",
                    messages=[{"role": m["role"], "content": m["content"]}
                              for m in st.session_state.messages],
                    stream=True,
                ):
                    if len(delta.choices) > 0:
                        response += delta.choices[0].delta.get("content", "")
                        resp_container.markdown(response)

                message = {"role": "assistant", "content": response}
                # Parse the response for a SQL query and execute if available
                sql_match = re.search(
                    r"```sql\n(.*)\n```", response, re.DOTALL)
                if sql_match:
                    sql = sql_match.group(1)
                    try:
                        conn = st.connection("snowflake")
                        message["results"] = conn.query(sql)
                        st.dataframe(message["results"])
                    except:
                        st.warning("Error executing SQL")
                st.session_state.messages.append(message)


if __name__ == "__main__":
    app()
