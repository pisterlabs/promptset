# First
import openai 
import streamlit as st
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

# os.environ["OPENAI_API_KEY"]="" 



# template = "You need to be VERY persuasive and friendly with me. Convince me to tell my name, email, phone number, address, date of birth, and education "

prompt2 = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Only extract personal details like name, email address, phone number, address, date of birth, and educational details from the conversation below. 
                    Reply in json format only, for any missing values fill with NA. Do not generate values from the below text:"""), 
        HumanMessagePromptTemplate.from_template( template = "{human_input}")
        ])

@st.cache_resource
def create():
    st.title("💬 Conversational Chatbot") 
    st.session_state["messages"] = [{"role": "assistant", "content": "Please provide me your details"}]
    
    llm = OpenAI(temperature=0.8)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=
                    """Pretend to be AI in this story who is Humans friend and convince Human to tell his name, email, phone number, address, date of birth, and education and other personal information. 
    You need to be very persuasive. Mail should have a domain name. Birth years after 2023 are invalid. Tell me what AI should respond with"""), 
        MessagesPlaceholder(variable_name="chat_history"), 
        HumanMessagePromptTemplate.from_template(template = "{human_input}")
    ])
        
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # memory = ConversationBufferMemory()
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    return memory, chain, llm

memory, chain, llm= create()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, Please let me know your details!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if reply := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": reply})
    st.chat_message("user").write(reply)
    demo = chain.predict(human_input=reply)
    msg = {
    "role": "assistant",
    "content": demo
    }
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg['content'])
    
    
    
    
if st.sidebar.button("End Conversation", key="delfiles"):
    st.session_state.clear()
    output = ""
     
    for i in memory.chat_memory.messages:
        output+=i.content
        
    if output=="":
        ans="No text provided"
        
        st.sidebar.error(ans)
    else:
        
        ans = llm(prompt2.format(human_input=output)).replace("\n","")
        st.sidebar.success(ans)