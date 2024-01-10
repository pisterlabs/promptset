from langchain.llms import VertexAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

text_llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=50,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
    streaming=True
)
tools = load_tools(["ddg-search"])
agent = initialize_agent(
    tools, text_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)

