import streamlit as st
import pandas as pd
from pytube import YouTube
from matplotlib.patches import Arc
from langchain.callbacks.manager import tracing_v2_enabled
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig
from openai.error import AuthenticationError
from langsmith import Client

from llm_stuff import (
    _DEFAULT_SYSTEM_PROMPT,
    get_memory,
    get_llm_chain,
    StreamHandler,
    get_langsmith_client,
)

# auth_key from secrets
auth_key = st.secrets['auth_key']

st.set_page_config(page_title='LLM with Streamlit', 
                   page_icon='👀', layout='centered', initial_sidebar_state='expanded' )

# to hide streamlit menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
# pass javascript to hide streamlit menu
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    if st.session_state.process_status == 'done':
        st.subheader('❓:red[ **Ask about the video**]')
        
        langsmith_project = None
        client = None
        instructions = "You are a helpful chatbot. Response questions considering the following text. If you don´t know the answer, say it and don´t create any new information. <text>"
        
        if st.session_state.openai_api_key.startswith("sk-"):
            system_prompt = instructions + st.session_state.chat_text.strip().replace("{", "{{").replace("}", "}}") + "</text>"
            memory = get_memory()

            chain = get_llm_chain(memory, system_prompt, 0)

            run_collector = RunCollectorCallbackHandler()

            def _get_openai_type(msg):
                if msg.type == "human":
                    return "user"
                if msg.type == "ai":
                    return "assistant"
                return msg.role if msg.type == "chat" else msg.type

            for msg in st.session_state.langchain_messages:
                streamlit_type = _get_openai_type(msg)
                avatar = "👀" if streamlit_type == "assistant" else None
                with st.chat_message(streamlit_type, avatar=avatar):
                    st.markdown(msg.content)

            if st.session_state.trace_link:
                st.sidebar.markdown(
                    f'<a href="{st.session_state.trace_link}" target="_blank"><button>Latest Trace: 🛠️</button></a>',
                    unsafe_allow_html=True,
                )

            def _reset_feedback():
                st.session_state.feedback_update = None
                st.session_state.feedback = None

            if prompt := st.chat_input(placeholder="Ask me a question!"):
                st.chat_message("user").write(prompt)
                _reset_feedback()

                with st.chat_message("assistant", avatar="👀"):
                    message_placeholder = st.empty()
                    stream_handler = StreamHandler(message_placeholder)
                    runnable_config = RunnableConfig(
                        callbacks=[run_collector, stream_handler],
                        tags=["Streamlit Chat"],
                    )
                    try:
                        if client and langsmith_project:
                            with tracing_v2_enabled(project_name=langsmith_project):
                                full_response = chain.invoke(
                                    {"input": prompt},
                                    config=runnable_config,
                                )["text"]
                        else:
                            full_response = chain.invoke(
                                {"input": prompt},
                                config=runnable_config,
                            )["text"]
                    except AuthenticationError:
                        st.error("Please enter a valid OpenAI API key.", icon="❌")
                        st.stop()
                    message_placeholder.markdown(full_response)

                    if client:
                        run = run_collector.traced_runs[0]
                        run_collector.traced_runs = []
                        st.session_state.run_id = run.id
                        wait_for_all_tracers()
                        url = client.read_run(run.id).url
                        st.session_state.trace_link = url
        else:
            st.error("Please enter a valid OpenAI API key.", icon="❌")
            st.stop()
    else:
        st.markdown('Process the video first!')
            
if __name__ == '__main__':
	main()   