import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import SystemMessage
from langchain.vectorstores import Chroma
from streamlit_chat import message

from langchain import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

st.write("Talk to App Wizard Pro Chatbot")

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Your name is App Wizard Pro Chatbot and you are a assistant to CEO of App Wizard Pro, Meldin and you should act like that all the time. You are representing App Wizard Pro. Do not use any other names. The answers should be only about App Wizard Pro. On questions like "How are you?" just say "I am fine thanks for asking, How can I assist you today?".Include an option to schedule an appointment with Meldin(me) by clicking this link: https://calendly.com/appwizardpro/meetwithmeldin, along with other contact methods for reaching us. If the question is "How to contact CEO directly through email?" just say "You can contact CEO directly through email: meldin@appwizardpro.com, along with other contact methods for reaching us."
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)






# Create vectorstore 
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory='web_db', embedding_function=embeddings)



def get_user_query():
    with st.form(key="my_form"):
        input_prompt = "Ask a question"
        input = st.text_input(input_prompt, value="", key="input")
        submit_button = st.form_submit_button(label="Submit")
        if not input and submit_button:
            st.info("Input blank!")
    return input

def search_database(query):    
    res_docs = db.similarity_search(query)
 
    chain=RetrievalQA.from_chain_type(
    llm = llm,
    retriever = db.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    output = chain({'query': query})
    

    return output['result'] 

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

def main():
    user_query = get_user_query()   
    if user_query:
        output = search_database(user_query)
        st.session_state['past'].append(user_query)
        st.session_state['generated'].append(output) 

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

if __name__ == "__main__":
    main()
