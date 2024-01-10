from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
import os
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone, Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import pinecone 
from langchain.llms import OpenAI
import pandas as pd
# initialize pinecone
pinecone.init(
    api_key="83af22f9-ec60-484a-8eba-3519c69b251f",
    environment="eu-west4-gcp",
    token=6000
)
#loader = UnstructuredExcelLoader("example_data/stanley-cups.xlsx", mode="elements")


# loader = UnstructuredExcelLoader(file_path="testDir\sampleincomestatement.xls",mode="elements")
# data = loader.load()
excel_path = "testDir\sampleincomestatement.xls"
xls = pd.ExcelFile(excel_path)

# Initialize the list to store tables
tables = []

# Loop through each sheet in the Excel file
for sheet_name in xls.sheet_names:
    # Read the sheet into a DataFrame
    df = pd.read_excel(xls, sheet_name)
    
    # Remove rows with null values
    df = df.dropna()
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Append the processed DataFrame to the tables list
    tables.append(df)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40) #chunk overlap seems to work better
documents = text_splitter.split_documents(tables)
print(len(documents))

index_name = "test-tangr2"
embeddings = OpenAIEmbeddings(openai_api_key='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')
vectorstore = Pinecone.from_documents(documents, embeddings, index_name=index_name)
#vectorstore = Pinecone.from_existing_index(index_name, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0,openai_api_key='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'), retriever)

chat_history = []
query = "show me a table of all compensation and benefits for 1 year "
result = qa({"question": query, "chat_history": chat_history})
print(result["answer"])

