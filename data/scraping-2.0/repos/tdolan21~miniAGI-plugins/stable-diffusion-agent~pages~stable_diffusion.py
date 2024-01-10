import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from gradio_tools.tools import (
    StableDiffusionTool,
    ImageCaptioningTool,
    StableDiffusionPromptGeneratorTool,
    TextToVideoTool,
)

from langchain.memory import ConversationBufferMemory
from gradio_tools.tools import StableDiffusionTool

tools = [
    StableDiffusionTool().langchain,
    ImageCaptioningTool().langchain,
    StableDiffusionPromptGeneratorTool().langchain,
    TextToVideoTool().langchain,
]

llm = OpenAI(temperature=0)
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools, llm, memory=memory, agent="conversational-react-description", verbose=True
)




st.title("miniAGI :computer:")
st.subheader("Stable diffusion plan and execute agent")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
        
        