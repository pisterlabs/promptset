import openai
import sys
import os
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import elastic_vector_search, pinecone, weaviate, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import re
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('API_key')
API_KEY = os.getenv('API_key')
def chatbot(user_input):
    llm = ChatOpenAI(api_key = API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial tax refund specialist. You will receive the values of: W2 Box 1, W2 Box 2,  W2 Box3, W2 Box4, W2 Box5, W2 Box6, and calculate the tax refund. Assume values of tax brackets in the USA in 2021.  Assume standard deduction of 12250. "),
        ("user", "Income: {income}, federal_taxes_withheld: {federal_taxes_withheld}, social_security_wage: {social_security_wage}, social_security_tax_withheld: {social_security_tax_withheld},  medicare_wages_and_tips: {medicare_wages_and_tips}, medicare_tax_withheld: {medicare_tax_withheld} ")
    ])
    output_parser = StrOutputParser()
    input_dict = {
        "income": user_input.get("income", ""),
        "federal_taxes_withheld": user_input.get("federal_taxes_withheld", ""),
        "social_security_wage": user_input.get("social_security_wage", ""),
        "social_security_tax_withheld": user_input.get("social_security_tax_withheld", ""),
        "medicare_wages_and_tips": user_input.get("medicare_wages_and_tips", ""),
        "medicare_tax_withheld": user_input.get("medicare_tax_withheld", "")
        }
    chain = prompt | llm | output_parser
    string_resp = chain.invoke(input_dict)
    string_resp=string_resp.replace(",","")
    last_dollar_index = string_resp.rfind('$')

    if last_dollar_index != -1:
        stripped_string = string_resp[last_dollar_index + 1:]
        number = re.findall(r'\d+', stripped_string)
        for i in number:
            if int(i) < int(user_input.get("income", "")) * 0.3 :
                return i
            else:
                return int(i) * 0.5
    else:
        return string_resp

    

memory = ConversationBufferWindowMemory(k = 20)


#conversation = ConversationChain(llm=llm)
def readPDF(user_input):
    pdf_reader = PdfReader('data/2023Test.pdf')
    output_parser = SimpleJsonOutputParser
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)



    embeddings = OpenAIEmbeddings()
    doc_search = FAISS.from_texts(chunks, embeddings)

    chain = load_qa_chain(ChatOpenAI(), chain_type="stuff")
    query = user_input
    docs = doc_search.similarity_search(query)
    return(chain.run(input_documents = docs, question = query))

numbers = re.findall(r'\d+', readPDF("Give me the value of Wage, Federal income tax withheld, Social security wage, social security tax withheld, medicare wages and tips, medicare tax withheld,  as single numbers seperated by commas"))


def convert_to_dict(numbers):

    income = numbers[0] 
    federal_taxes_withheld = numbers[1] 
    social_security_wage = numbers[2]
    social_security_tax_withheld = numbers[3]
    medicare_wages_and_tips = numbers[4]
    medicare_tax_withheld = numbers[5]

    user_input_dict = {
            "income": income,
            "federal_taxes_withheld": federal_taxes_withheld,
            "social_security_wage": social_security_wage,
            "social_security_tax_withheld": social_security_tax_withheld,
            "medicare_wages_and_tips":  medicare_wages_and_tips,
            "medicare_tax_withheld": medicare_tax_withheld
        }
    return user_input_dict

