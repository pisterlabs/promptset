import openai
import chromadb
import os

from typing import Optional
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain, 
)
from langchain.document_loaders import PyPDFLoader

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


#GPT
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#pydantic
class get_table_of_contents(BaseModel):
    company_name :str = Field(..., discription="Campany name")
    bussiness :bool = Field(..., discription="Does the document include the business or not?")
    average_annual_income :bool = Field(..., discription="Does the document contain data on average annual income or not?")
    
#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world class algorithm for extracting information in structured formats."),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)
    

#vectorstore, retriever
openai_ef = OpenAIEmbeddings()
persist_path = "/home/kazuki/study/Study/4/Chroma"

##load and split
pdf_path = "/home/kazuki/study/Study/4/demoproject/media/NTTdata--extract(1)_aZwiSOf.pdf"
loader = PyPDFLoader(f"{pdf_path}")
pages = loader.load_and_split()

print(f"pages:{pages}\n")

##making collection
#collection = Chroma(collection_name="get_a_table_of_contents", embedding_function=openai_ef, persist_directory=persist_path)
#vectorstore = collection.from_documents(pages,embedding=openai_ef, collection_name="get_a_table_of_contents", persist_directory=persist_path)
#retriever = vectorstore.as_retriever()


##for a collection which was created before
client = chromadb.PersistentClient(path=persist_path)
vectorstore = Chroma(collection_name="get_a_table_of_contents", client=client, embedding_function=openai_ef)
retriever = vectorstore.as_retriever()

##compressing
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter,DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=".")
redundant_filter = EmbeddingsRedundantFilter(embeddings=openai_ef)
relevant_filter = EmbeddingsFilter(embeddings=openai_ef, similarity_threshold=0.76)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("Information that constitutes the company, such as company name and business activities")
chain_input = f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(compressed_docs)])

#chain
chain = create_structured_output_chain(get_table_of_contents, llm, prompt, verbose=True)
print("beforechain\n")
print(chain_input)
#table_of_contents = chain.run(f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(compressed_docs))
table_of_contents = chain.run(chain_input)

print(table_of_contents)