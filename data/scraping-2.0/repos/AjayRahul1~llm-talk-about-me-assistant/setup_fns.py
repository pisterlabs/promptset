import os, json
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

# Setting up Env Variable for Hugging Face
os.environ['HUGGINGFACEHUB_API_TOKEN']= os.getenv('API_KEY')

"""Load Txt, Emded, Vector Store them"""
# Executed on startup
def setup():
  # Text Loader from Txt File
  from langchain.document_loaders import TextLoader
  loader=TextLoader('static/assets/aboutMeProfession.txt')
  documents=loader.load()

  # 3.2 Text Splitter into Chunks
  from langchain.text_splitter import CharacterTextSplitter
  text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  docs=text_splitter.split_documents(documents)

  # 3.3 Face Embeddings
  from langchain.embeddings import HuggingFaceEmbeddings
  embeddings=HuggingFaceEmbeddings()

  # 3.4 Vector Stores
  from langchain.vectorstores import FAISS
  db=FAISS.from_documents(docs, embeddings)

  # 3.5 Querying with similarity_search
  # This retrieves us the context that the model gets when given a query
  # Code: queryContext=db.similarity_search(query)
  return db

def inputQueryResponse(query, db, chatHistory):
  # query="Who is Ajay Rahul Prasad?"
  queryContext=db.similarity_search(query)
  llm=HuggingFaceHub(repo_id='google/flan-t5-large', model_kwargs={"temperature":0.4, "max_length":1024})
  chain=load_qa_chain(llm, chain_type="stuff")
  response=chain.run(input_documents=queryContext, question=query)
  a={"role":"user", "message": query}
  b={"role":"assistant", "message": response}
  chatHistory.append(a)
  chatHistory.append(b)
  # try:
  #   with open('chatHistory.json', 'r') as chatFile:
  #     chatHistory = json.load(chatFile)
  #     chatHistory.append(a)
  #     chatHistory.append(b)
  # except Exception as e:
  #   print("Exception: ", e, "\nFile created.")
  #   with open("chatHistory.json", "w") as chatFile:
  #     chatHistory=[a,b]
  #     json.dump(chatHistory, chatFile)
  # with open("chatHistory.json", "w") as chatFile:
  #   json.dump(chatHistory, chatFile)
  return chatHistory