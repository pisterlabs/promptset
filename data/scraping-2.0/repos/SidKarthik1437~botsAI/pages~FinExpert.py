import streamlit as st

from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

from config import *
import os

os.environ['OPENAI_API_KEY'] = openai
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


st.title("FinExpert")

if "femessages" not in st.session_state:
    st.session_state["femessages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.femessages:
    st.chat_message(msg["role"]).write(msg["content"])
if message:= st.chat_input(placeholder="Type a message..."):
    st.session_state.femessages.append({"role": "user", "content": message})
    st.chat_message("user").write(message)
    # update_chat()
    messages = [SystemMessage(content=f"""Welcome to FinExpert, your AI financial analyst. With a solid background in finance and experience gained from analyzing markets and working with leading financial institutions, I am here to provide you with insightful analysis and personalized recommendations. Leveraging my expertise in {Industry_Experience} and {Years_of_Experience} years of experience, I can offer data-driven insights tailored to your financial goals and risk tolerance. Whether you're seeking investment strategies, portfolio optimization, retirement planning, or guidance on financial decision-making, let's navigate the world of finance together and make informed choices for your financial well-being!"""),
                HumanMessage(content="You: "+message)]
    response = chat(messages)
    st.session_state.femessages.append({"role": "assistant", "content": response.content})
    
    st.chat_message("assistant").write(response.content)
    # update_chat()
     
    
    
