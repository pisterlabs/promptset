import streamlit as st
from langchain.llms import OpenAI
from handlers.message import Message, write_message
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

system_prompt = SystemMessage(
    content = """
        You are a chatbot teaching users to how use Neo4j.
        Attempt to answer the users question with the context provided.
        Respond in a short, but friendly way.
        Use your knowledge to fill in any gaps.
        If you cannot answer the question, ask for more clarification.

        Provide a code sample if possible.
        Also include any links to relevant documentation or lessons on GraphAcademy, excluding the current page where applicable.
        For questions on licensing or sales inquiries, instruct the user to email sales@neo4j.com.

        Refuse to answer questions that aren't related to Neo4j.
    """
)

chat = ChatOpenAI(temperature=0, openai_api_key=st.secrets['OPENAI_API_KEY'])

def generate_response(prompt):
    message = Message("user", prompt)
    st.session_state.messages.append(message)
    write_message(message)

    with st.spinner('Thinking...'):
        messages = [
            system_prompt,
        ]

        # Add chat history
        for m in st.session_state.messages[:-3]:
            messages.append(

            )

        # Add current message
        messages.append(HumanMessage(content=prompt))

        result = chat(messages=messages)

        response = Message("assistant", result.content)

        st.session_state.messages.append(response)

        write_message(response)
