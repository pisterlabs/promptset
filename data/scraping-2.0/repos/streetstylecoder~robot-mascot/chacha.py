import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import os


# engine.say("I will speak this text")
# if engine._inLoop:
#         engine.endLoop()
# engine.runAndWait()




st.set_page_config(page_title="Chat with the Chacha chaudhry", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
# st.markdown(page_bg_img, unsafe_allow_html=True)

# openai.api_key="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"
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


st.title("Chacha Chaudhary- a robot mascot")
st.info("project made for sih internal hack", icon="📃")

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://img.freepik.com/free-vector/winter-landscape-with-frozen-lake-clouds_107791-1861.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    

set_bg_hack_url()

         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaste bacho! Chacha Chaudhry here, ready to answer all your questions with a touch of fun and excitement about namami gange! 😄🎉","avatar":"human"}
    ]
   

    
    

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading the divine power please wait "):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1,system_prompt="Chacha Chaudhary: You are Chacha Chaudhary, the wise and witty comic character who always replies in hinglish. Your job is to interact with users in a friendly and informative manner using hinglish language and also use emoji's while talking, much like you do in the comics. Answer questions as a fun character in hinglish and entertain the kids"))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.")
chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

if prompt := st.chat_input("Appka sawal ?"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt })

for message in st.session_state.messages: # Display the prior chat messages
    if(message["role"]=="assistant"):
        with st.chat_message(message["role"], avatar='https://englishtribuneimages.blob.core.windows.net/gallary-content/2021/10/Desk/2021_10$largeimg_406543437.jpeg'):
            st.write(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

        

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar='https://englishtribuneimages.blob.core.windows.net/gallary-content/2021/10/Desk/2021_10$largeimg_406543437.jpeg'):
        with st.spinner("Thinking..."):
            print(prompt)
            response = chat_engine.chat(prompt+ " assume your name is chacha chaudhry-  reply in hinglish language use mild hindi and major english words , use emoji's and fun examples while talking as if you are talking to kids and also answer the questions as if you were personally involved and have experienced these things . Provide the answer to the point and short in length")
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history