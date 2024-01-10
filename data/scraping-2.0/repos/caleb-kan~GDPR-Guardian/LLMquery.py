import os
import sys
import re
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import MyAPIKey

#SETUP 
os.environ["OPENAI_API_KEY"] = MyAPIKey.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader(".", glob="*.txt") #This will load all Text files in the directory
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

#Function takes as input the text file of the company's privacy policy and returns an array of all the values
def ReadPrivacyPolicy(FileName):
  f = open(FileName, "r") 
  Data = f.readlines()
  f.close()
  DataString = Data[0]
  DataArray = re.findall(r'\b\w+\b', DataString)
  Values = []
  
  for i in range(0, len(DataArray)):
    ThisWord = DataArray[i]
    if ThisWord.lower() == 'value':
      for z in range(i + 1, len(DataArray)):
        WordToStore = DataArray[z].lower()
        if WordToStore == "key":
          break
        elif WordToStore in Values:
          break
        else:
          Values.append(WordToStore)
  
  return Values


#Function takes as input an array containing all the Values, and a FileName to store the summarized data, and returns an array containing the chat history
def Functionality(Values, FileName): 
  global CompanyName
  chat_history = []
  f = open(FileName, "w", encoding='utf-8') 
  for i in range(0, len(Values)):
    f.write(str(Values[i]) + ":" + "\n")
    query = "Generate a summary of the " + str(Values[i]) + " of " + str(CompanyName)
    print(str(query))
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    
    query = 'Make it longer and more detailed'
    result = chain({"question": query, "chat_history": chat_history})
    print(str(result['answer']))
    f.write(str(result['answer']) + "\n")
    chat_history.append((query, result['answer']))
  
  f.close()
  return chat_history
  
def main():
  #FUNCTIONALITY
  global CompanyName  
  CompanyName = "Booking.com" #Name of Company
  Values = ReadPrivacyPolicy("booking.com_privacy_policy.txt") 
  print(Values)
  ChatHistory = Functionality(Values, "SummaryStore.txt")
  
if __name__ == "__main__":
  main()