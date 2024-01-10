from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st

st.set_page_config(page_title="Local Model Chat", page_icon="📖")

# streamlit picker for choice of LLM model
llm_picker = st.sidebar.selectbox("Choose an LLM model", [ "Mistral", "llama2"])
st.title(f"📖 {llm_picker} Chat")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Select the llm 
# llm = Ollama(model="mistral:latest")

if llm_picker == "llama2":
    llm = Ollama(model="llama2:latest")
else:
    llm = Ollama(model="mistral:latest")

template_text = st.sidebar.text_area("Template", value="You are an AI chatbot having a conversation with a human.", height=500)

# Set up the LLMChain, passing in memory
template = """
{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template_text + template)

print(f"History: prompt={prompt}, memory={memory}")

llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = llm_chain.run(prompt)
    st.chat_message("ai").write(response)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    st.write(msgs.messages)
    view_messages.json(st.session_state.langchain_messages)