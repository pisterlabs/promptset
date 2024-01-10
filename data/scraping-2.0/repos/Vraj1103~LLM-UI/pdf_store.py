from pathlib import Path
from llama_hub.file.pdf.base import PDFReader
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

# Load documents from PDF
loader = PDFReader()
def storePDF():
    documents = loader.load_data(file=Path("./path_to_your_pdf.pdf"))

    # Create an index with the document
    index = VectorStoreIndex(documents)

    # Instantiate a Retriever and a Query Engine
    retriever = RecursiveRetriever(index)
    qe = RetrieverQueryEngine(llm=OpenAI(model="gpt3-large"), retrievers=[retriever])

    # create a context for the service
    ctx = ServiceContext()   
    # print(qe.ask({"text": msg}, ctx=ctx))
    return qe, ctx

# print("You can now chat with the system. Type 'quit' to stop.")
  
# while True:
#     msg = input("You: ") 
#     if msg.lower() == 'quit':
#         break