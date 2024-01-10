import openai
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAIChat
import faiss
import os
import pickle
from langchain.chains import ConversationalRetrievalChain
import datetime

from dotenv import load_dotenv
import os

# load_dotenv('question_answer_version02/.env')
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


index = faiss.read_index("question_answer_version02/vector_data/docs.index")
# index = faiss.read_index("vector_data/docs.index")
with open("question_answer_version02/vector_data/faiss_store.pkl", "rb") as f:
# with open("vector_data/faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
openai.api_key = os.environ["OPENAI_API_KEY"]
message_history = ConversationBufferMemory(memory_key="chat_history")
template = """
As the internal chatbot assistant of Metastar, you have access to the following chat history: {chat_history}.
When a user inputs a query, you must search for the relevant information and provide a suitable conclusion.
If no information is found, please return an empty string.

Human: {human_input}
Chatbot:
"""
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)
# hack will be optimised soon

message_history.chat_memory.add_ai_message("what is metastar?")
message_history.chat_memory.add_ai_message("hello!")
message_history.chat_memory.add_ai_message("Hi there! I'm the internal chatbot assistant of Metastar. How can i help you today?")
message_history.chat_memory.add_ai_message("Metastar is an integrated real estate information provider platform that quickly provides real estate related documents")
message_history.chat_memory.add_user_message("What is metastar web site url?")
message_history.chat_memory.add_ai_message("metastarglobal.io")


# object initilization 
llm = OpenAIChat(model_name="gpt-3.5-turbo")
doc_CONDENSE_QUESTION_PROMPT="you will find the relavent information from question, keep in mind question may be ask in different language so translate question first in english then make answer"
get_aks_chain = ConversationalRetrievalChain.from_llm(llm, retriever=store.as_retriever(search_type="similarity",
                                                                                        search_kwargs={"k": 2},))
open_template = """You are an Internal Chatbot Assistant of Metastar. You can answer questions based on the 
    chat history, company and apartment_price documents. also use the language the question is asked in.
    try to make your answer concise and informative if you find any price in apartment_price docuemnts show to user what you find similar to user query if user ask price for apartment.


    Question: {question} Chat history: {chat_input} Company documents info: {docs_info} , apartment_price documents:{aps} , Answer:
    """
prompt_temp = PromptTemplate(
    template=open_template,
    input_variables=["question", "docs_info", "chat_input","aps"]
)
gpt_llm_chain = LLMChain(prompt=prompt_temp, llm=llm)

memory_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=message_history,)



def get_conversation_history(user_input):
    response = memory_llm_chain.predict(human_input=user_input)
    message_history.chat_memory.add_user_message(user_input)
    return response



def get_aks_doc(query):
    result = get_aks_chain({"question": query, "chat_history": []})
    return result['answer']

def chatgpt(question, history_output, docs_output,aps):
    return gpt_llm_chain.predict(question=question, docs_info=docs_output, chat_input=history_output,aps=aps)

# @tool('ms_qa',return_direct=True)
from langdetect import detect
def translate(response,question):
    question_lang=detect(question)
    response_lang=detect(response)
    if question_lang==response_lang:
        return response
    else:
        template=f"please translate this text:<{response}> into <{question_lang}> language."
        ouput=llm.predict(template)
        return ouput



def predict(text):
    try:
        docs_output = get_aks_doc(text)
        history_output = get_conversation_history(text)
        response = chatgpt(text, history_output, docs_output,aps)
        response=translate(response,text)
        return response
    except Exception as e:
        with open("question_answer_version02/error_log.txt","a") as f:
            f.write(str(e))
            f.write(f"time: {datetime.datetime.now()} \n")
        response="Sorry, this link is temperorily unavailable due to traffice but we will be available soon."
        response=translate(response,text)
        return response



