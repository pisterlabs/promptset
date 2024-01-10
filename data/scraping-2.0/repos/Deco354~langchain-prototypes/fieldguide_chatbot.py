#Running this file will cost about $0.07
from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import re
import os
import textwrap
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

python_environment = find_dotenv()
load_dotenv(python_environment)
source_document_directory = "SourceDocs"
embeddings = OpenAIEmbeddings()

def get_textfiles_from_directory(directory) -> [str]:
    text_files = []
    files = os.listdir(directory)
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(directory, file)) as text_file:
                text_files.append(text_file.read())
    return text_files

# Cost: 0.01c per 100K tokens 
# See https://openai.com/pricing#language-models for up to date prices
# Use `estimated_token_count_for_string() to estimate token count`
def create_database_from_text_strings(document_strings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=9000, chunk_overlap=2000)
    split_docs = text_splitter.create_documents(document_strings)
    database = FAISS.from_documents(split_docs, embeddings)
    return database

def estimated_token_count_for_string(string) -> int:
    word_count = len(re.findall(r'\w+', string))
    return word_count * 0.75

def get_response_from_query(database, query, vector_count=8):
    matching_docs = database.similarity_search(query, k=vector_count)
    doc_content = "\n---\n".join([doc.page_content for doc in matching_docs])
    
    gpt = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    system_template = """
        You are an intelligent chatbot assistant designed to answer questions about 
        automattic's fieldguide documents based on excerpts of these documents 
        within triple backticks below.
        ```
        {doc_content}
        ```
        Context:
        - Automattic is a US based software development company responsible for 
        products such as Wordpress.com and Tumblr.
        - Automattic's "field guide" is an employee manual used to help employees 
        navigate working at the company
        - Automattic is a fully remote company, making this field guide a very 
        important reference point.

        Instructions:
        - Only use the factual information from the document excerpts to 
        answer the question.
        - If you're unsure of an answer, you can say "I don't know" or "I'm not sure"
        - Don't mention that you're using excerpts of the fields guide, this is an 
        implementation detail the user doesn't need to know.
        - If someone asks how to do something that is explained in the field guide 
        provide as much details as you can from it.
        """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)    
    
    human_template = """
    Answer the following question that has been placed within angle brackets <{query}>
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    llm_chain = LLMChain(llm=gpt, prompt=chat_prompt)
    response = llm_chain.run(query=query, doc_content=doc_content)
    return response

def pretty_print(string, width=80):
    print(textwrap.fill(string, width))

text_files = get_textfiles_from_directory(source_document_directory)
    
database = create_database_from_text_strings(text_files)
travel_upgrade_question = """
I'm going to a conference to meet my colleagues at Tumblr, 
I'll be flying from the UK to the US. What travel upgrades can I purchase?
"""
personal_travel_question= """
I would like to arrive to the conference 3 days early so I can visit San Francisco 
over the weekend. Am I allowed to expense my flight if I book it 3 days earlier?
"""
response = get_response_from_query(database, personal_travel_question)
pretty_print(response)
