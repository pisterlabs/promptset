from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# import os 
# import pickle


def pdf_reader(pdf, open_ai_key):
    pdf_read = PdfReader(pdf)
    text = ""
    for page in pdf_read.pages:
        text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)  
    # store_name = pdf.name[:-4]
    # if os.path.exists(f"{store_name}.pkl"):
    #     with open(f"{store_name}.pkl", "rb") as f:
    #         db = pickle.load(f)
    # else:
    embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key)  
    db = FAISS.from_texts(chunks, embeddings)  
        # with open(f"{store_name}.pkl", "wb") as f:
        #     pickle.dump(db, f)      
    return db
    
def get_query_response(db, query, open_ai_key, k=4):
    docs = db.similarity_search(query, k=4)
    merged_docs = " ".join([doc.page_content for doc in docs])
    llm = OpenAI(openai_api_key = open_ai_key, model = "gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""you are a helpful PDF assistant whos job is to answer questions
                    based on the text document provided. 
                    Answer the following question: {question}
                    by searching through the following document: {docs}
                    only use factual informtion from the transcript. 
                    if you feel you don't have sufficient information
                    to answer the question, say "i don't know" your answer
                    should be verbose and detailed.""",
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    response = chain.run(question=query, docs = merged_docs)
    response = response.replace("\n", " ")
    return response
    

