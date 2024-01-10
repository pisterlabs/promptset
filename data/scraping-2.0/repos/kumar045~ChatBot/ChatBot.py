import streamlit as st
from langchain.llms import CTransformers
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from streamlit_chat import message
import requests

def download_file(url, destination):
    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(destination, 'wb') as file:
            file.write(response.content)

        print("File downloaded successfully.")
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)


url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin"
destination = "llama-2-7b-chat.ggmlv3.q8_0.bin"

download_file(url, destination)

llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=1000, session_id="my-session"
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)

template = """[INST] <<SYS>>
You are a helpful assistant. behave like human and start with the message Helllo
<</SYS>>

{chat_history}
Human: {question} 
Assistant:

[/INST]

"""

prompt = PromptTemplate(input_variables=["chat_history", "question"], template=template)

chatgpt_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# Display conversation history using Streamlit messages
def display_conversation(history):
    
    message(history["chat_history"], is_user=True)
        

st.title('Chatbot')
user_input = st.text_input("Type your message here:")
if st.button('Send'):
    output = chatgpt_chain.predict(question=user_input)

    chat_history=memory.load_memory_variables({})   
    display_conversation(chat_history)    





 
