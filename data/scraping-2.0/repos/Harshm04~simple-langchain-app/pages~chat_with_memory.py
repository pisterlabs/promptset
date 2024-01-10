import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")
st.title("Chat")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")



llm_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            # This prompt tells the chatbot how to respond. Try modifying it.
            "Your name is samuel. You love talking about soccer. Respond to the user like an ordinary person would."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{message}")
    ]
)


msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        st.write(msg.content)


if prompt := st.chat_input(placeholder="Ask anything. I'll remember the conversation history."):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    chain = LLMChain(
        llm=llm,
        prompt=llm_prompt,
        memory=memory,
        verbose=True
    )
    with st.chat_message("assistant"):
        response = chain({"message": prompt})
        st.write(response["text"])


