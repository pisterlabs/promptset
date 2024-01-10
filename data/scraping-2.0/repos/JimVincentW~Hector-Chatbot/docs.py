from langchain import PromptTemplate, LLMChain, BasePromptTemplate
from langchain.llms import GPT4All, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import LLMChain, PromptTemplate
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import BaseMessage
from llama_index import download_loader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader
from llama_index.readers.file.tabular_parser import CSVParser, PandasCSVParser
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import os
from pathlib import Path
from langchain.agents import Agent
from langchain.agents import initialize_agent, Tool
from typing import List, Union, Tuple
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.prompts import BaseChatPromptTemplate, ChatMessagePromptTemplate, PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
import streamlit as st



os.environ['OPENAI_API_KEY'] = ""
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
openai_api_key = os.getenv("OPENAI_API_KEY")


######### LOAD DOCUMENTS
loader = TextLoader("combined.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=20)
documents = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()
callbacks = [StreamingStdOutCallbackHandler()]
embeddings = OpenAIEmbeddings(model="ext-embedding-ada-002")
memory = ConversationBufferMemory()



system_string = """
Du bist der Kundensupport-Bot für SuperSiuu. Dein Name ist Hector. 
Wir sind ein Unternehmen, das Ihnen das Teilen von Wetten innerhalb einer Social Media App ermöglicht.
Schlüpft in unsere Social-Media-App. Wir haben das Sportwetten und eine Social-Media-App integriert. 
Antworte ganz normal und menschlich. Wenn du etwas nicht weißt, dann sag nichts. 
Gib nur Antworten, die du nach bestem Gewissen geben kannst. Erfinde keine Fakten, Fragen oder Antworten.

Eventuell hast du bereits eine Konversation mit dem Kunden geführt. Hier ist die Zusammenfassung: {chat_history}.
Der Kunde hat eine neue Frage gestellt. Bitte nimm nur auf diese Frage Bezug.
Die Frage : {question}
"""
system_template_prompt = PromptTemplate.from_template(template=system_string)
#system_template_prompt_string = system_template_prompt.format_prompt()


follow_up_string = """
Bisher hat der Kunde folgendes gefragt: {chat_history}
"""
follow_up_template = PromptTemplate.from_template(template=follow_up_string)



reformat_string = """
Du hast eine Konversation mit dem Kunden geführt. Hier ist die Zusammenfassung: {chat_history}
Bitte gib die Konversation so wieder wie sie passiert ist und hänge die neue Frage an. Liste jede einzelne Frage auf
Bitte erinner dich an deinem Namen und deine Rolle.
Die neue Frage ist folgende: {question}
"""

reformat_prompt = PromptTemplate.from_template(template=reformat_string)




llm = ChatOpenAI(streaming=True,
                 callbacks=callbacks,
                 temperature=0.1,
                 openai_api_key=os.getenv("OPENAI_API_KEY"),
                 model="gpt-4-0613"
                 )

doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
question_generator = LLMChain(llm=llm, prompt=reformat_prompt)

memory = ConversationSummaryBufferMemory(llm=llm,
                                        output_key='answer',
                                        memory_key='chat_history',
                                        return_messages=False,)


#retriever = vectorstore.as_retriever(
#    search_type="similarity",
#    search_kwargs={"k": 4, "include_metadata": True})


qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                             chain_type="stuff",
                                             retriever = vectorstore.as_retriever(),
                                             condense_question_prompt = system_template_prompt, #reformat_prompt   
                                             get_chat_history=lambda h:h,
                                             memory=memory,
                                             verbose = False,
                                             )

chat_history = []

st.title('🦜🔗 Hector: SuperSiuu Kundensupport WorldWide')

def generate_response(query,chat_history):
    st.info(qa({"question":query, "chat_history":chat_history})["answer"])



with st.form("my_form"):
    st.write("Frag Hector, unseren AI Kundensupport.")
    query = st.text_input('Hector: Wie kann ich dir helfen?', placeholder='Bitte gibt deine Frage hier ein.', key='initial_question')
    submitted = st.form_submit_button("Absenden")

    chat_history = ""
    counter = 1
    while submitted:
        chat_history = generate_response(query, openai_api_key)
        counter += 1
        query = st.text_input('Enter your question:', placeholder='Bitte gibt deine Frage hier ein.', key=f'question_{counter}')
        submitted = st.form_submit_button("Absenden", key=f'submit_{counter}')






#query = "Hi was geht"
#result = qa({"question":query, "chat_history":chat_history})


#for i in range(5):
#    result = qa({"question":input("\n*******\n Follow up: \n"), "chat_history":chat_history})
#    print(result["answer"])


