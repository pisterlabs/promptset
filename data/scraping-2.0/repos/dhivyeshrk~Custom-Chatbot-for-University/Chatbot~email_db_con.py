import streamlit as st
import os
from langchain.vectorstores import Chroma
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer, util
import os
import re
import sqlite3
import smtplib
from email.message import EmailMessage

os.environ['OPENAI_API_KEY'] = ''

# Set Streamlit page configuration
st.set_page_config(
    page_title='User - LLM QA File',
    page_icon=":information_desk_person:",
    menu_items=None
)
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k}, metadata_fields=['purpose'])
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

def get_similar_use_case(query: str):
    root_dir = r"Similar_check_TextFiles"
    file_names = os.listdir(root_dir)
    allscore = []
    for file_name in file_names:
        file_path = os.path.join(root_dir, file_name)
        with open(f"{file_path}", 'r') as f:
            text = f.read()
        sentences = [sentence.strip() for sentence in re.split(r'[.!?]', text) if sentence.strip()]
        mscore = -10
        for sen in sentences:
            embed1 = model.encode(sen, convert_to_tensor=True)
            embed2 = model.encode(query, convert_to_tensor=True)
            cosine_score = util.pytorch_cos_sim(embed2, embed1)
            mscore = max(mscore,cosine_score)
        allscore.append([mscore,file_name])
    temp = [i for i,j in allscore]
    result = [[msc, fname] for msc, fname in allscore if msc == max(temp)]
    return result[0]

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

if __name__ == "__main__":
    st.subheader('LLM Chatbot Application :information_desk_person:')
    vector_store = Chroma(persist_directory=r"C:\Users\DELL\DataspellProjects\Chatbot_Test_2\ChatWithDocument\Data_Files",
                          embedding_function=OpenAIEmbeddings())
    st.session_state.vs = vector_store
    q = st.text_input('Ask a question about the content of your file:')
    if q and get_similar_use_case(q)[0] > 0.5:
        email_type = get_similar_use_case(q)[1]
        conn = sqlite3.connect('Mail_Feature.db')
        cursor = conn.cursor()
        yn1 = st.text_input('Do you want to send an email ? ')
        if yn1 == 'yes':
            cursor.execute(f"Select REQUIRED_PARAMETER1,REQUIRED_PARAMETER2,REQUIRED_PARAMETER3 from Level1 where TYPE_OF_QUERY = '{email_type}'")
            Required_parameters = cursor.fetchall()
            name = st.text_input('Name: ')
            roll = st.text_input("RollNo: ")
            cursor.execute(f"SELECT DESTINATION_MAIL1, DESTINATION_MAIL2 FROM LEVEL1 WHERE TYPE_OF_QUERY = '{email_type}'")
            destination_mail = cursor.fetchone()
            if destination_mail[0]=='FACULTY':
                fac_mail = st.text_input("Enter faculty emails: ")
            entered_password = st.text_input('Enter Password')









