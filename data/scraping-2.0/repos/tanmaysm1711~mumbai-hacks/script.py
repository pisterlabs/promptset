from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers, AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.agents import create_csv_agent
import sys
import os
import pandas as pd

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://notesocean.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "ad09dd1c72e24406ae47cf059d760aa3"

df = pd.read_csv('./data/inventory_data.csv')

# DB_FAISS_PATH = "./vectorstore/db_faiss"
loader = CSVLoader(file_path="data/inventory_data.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
# print(data)
#
# # # Print the colors of each car
# for row in data:
#     print(row)
#
# # Split the text into Chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
# text_chunks = text_splitter.split_documents(data)
#
# for text in text_chunks:
#     print(f'text_chunks --> {text}')
#
# print(f'Number of text chunks --> {len(text_chunks)}')
#
# # Download Sentence Transformers Embedding From Hugging Face
# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# print(f'Embeddings --> {embeddings}')
#
# # Converting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
# docsearch = FAISS.from_documents(text_chunks, embeddings)

# docsearch.save_local(DB_FAISS_PATH)

# llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
#                     model_type="llama",
#                     max_new_tokens=512,
#                     temperature=0.2)

# Create an index using the loaded documents
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

# Create an instance of Azure OpenAI
# Replace the deployment name with your own
llm = AzureOpenAI(
    deployment_name="social-media-app",
    model_name="gpt-35-turbo",
    temperature=0.5,
)

# agent = create_csv_agent(llm, './data/inventory_data.csv')

# Create a question-answering chain using the index
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

# print(f'This is docsearch.as_retriever() --> {docsearch.as_retriever()}')
#
# qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

while True:
    chat_history = []
    # query = "What is the value of  GDP per capita of Finland provided in the data?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    # query += '\nGive only the relevant answer and no other information.'
    # result = agent.run(query)

    result = chain.run(query)
    chat_history.append(query)
    chat_history.append(result['response'])

    print("Response: ", result['response'])
