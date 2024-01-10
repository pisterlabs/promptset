from fastapi import FastAPI, File, UploadFile
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os
import io
from pymongo import MongoClient
import uvicorn

app = FastAPI()
os.environ["OPENAI_API_KEY"] = " Enter your API KEY"

# initialize MongoDB python client
client = MongoClient("mongodb+srv://baravind031:SznII1KJRhASObXl@cluster0.uejwn03.mongodb.net/I")
db_name = "lanchain_db"
collection_name = "langchain_col"
collection = client[db_name][collection_name]
index_name = "langchain_demo"

response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    # ResponseSchema(name="source", description="source is used to answer the user's question from should be from context.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

# Define the prompt template
template = """use the following pieces of context to answer the question at the end. If you don't know the answer, just say that this question is not related to context, don't try to make up an answer out of the provided context, your purpose is to provide information to user question in detail from Pdf texts  .
\ncontext:{context}\nQuestion:{question}\n{format_instructions}"""

human_template = "give the answer from the pdf, please indicate if you are not sure about answer. Do NOT Makeup."
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(template),
        HumanMessagePromptTemplate.from_template(human_template)],
    input_variables=['context', "question"],
    partial_variables={"format_instructions": format_instructions}
)
embeddings = OpenAIEmbeddings()



@app.get("/")
async def root():
   return {"message": "loding..."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # file upload 
    pdf_data = await file.read()


    reader = PdfReader(io.BytesIO(pdf_data))
   
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()
    
    # Split the raw text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    
 
    docsearch = MongoDBAtlasVectorSearch.from_documents(texts, embeddings, collection=collection, index_name=index_name)
    
    
    return {"message": "File uploaded successfully."}

@app.post("/askquestion")
def ask_question(question: str):
    docsearch = MongoDBAtlasVectorSearch(
    collection, embeddings, index_name=index_name)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    chain_type_kwargs = {"prompt": prompt}
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=docsearch.as_retriever(), chain_type="stuff", chain_type_kwargs=chain_type_kwargs,verbose=True)
                                            # return_source_documents = True,)
    
    if qa_chain is None:
        return {"message": "Please upload a file first."}
    
     # Question answering
    response = qa_chain.run({"query": question, "format_instructions": format_instructions,})
    
    if response:
        return {"answer": response}
    else:
        return {"message": "The question is not related to the PDF content."}
 

if __name__ == "__main__":
    uvicorn.run("fast_api_QA:app", host="0.0.0.0", port=8000)


# # uvicorn fast_api_QA:app --reload


 



