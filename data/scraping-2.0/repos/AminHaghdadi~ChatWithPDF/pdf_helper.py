
from langchain.document_loaders import PyPDFLoader
from keys import openai_API
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
# Import os to set API key
import os

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY']=openai_API



def chat_with_your_pdf(pdf_file,query) :
    # Save the uploaded file to a temporary file
    if not os.path.exists("temp"):
        os.makedirs("temp")
    with open(os.path.join("temp", "uploaded.pdf"), "wb") as f:
        f.write(pdf_file.getbuffer())
    file_path = os.path.join("temp", "uploaded.pdf")
    # Simple method - Split by pages

    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    # print(pages[0])

    # SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
    chunks = pages

    # Get embedding model
    embeddings = OpenAIEmbeddings()

    # Create vector database
    db = FAISS.from_documents(chunks, embeddings)


    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    query = str(query)
    docs = db.similarity_search(query)

    return chain.run(input_documents=docs, question=query)











