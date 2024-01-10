from langchain.chains.summarize import load_summarize_chain, _load_stuff_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader, GitLoader
import constants
import os

os.environ["OPENAI_API_KEY"] = constants.APIKEY

while True: 

    command = int(input("\nSelect:\n1- PDF File.\n2- WebPage.\n3- TXT File.\n0 To Exit ==> "))
  
    if command == 1:
        filename = input("Your PDF File Name ==> ")
        file_path = os.path.join("Data/", filename)
        loader = PyPDFLoader(file_path) # PDF File Summerizer
    if command == 2:
        WebPage_link = input("WebPage link ==> ")
        loader = WebBaseLoader(WebPage_link) # WebPage Summerizer

    if command == 3:
        filename = input("Your TXT File Name ==> ")
        file_path = os.path.join("Data/", filename)
        loader = TextLoader(file_path) # TXT File Summerizer
    if command == 0:
        break

    docs = loader.load()

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")

    summarization_chain = load_summarize_chain(llm, chain_type="stuff")

    summarized_result = summarization_chain.run(docs)

    print("Summarized Text:\n", summarized_result)
