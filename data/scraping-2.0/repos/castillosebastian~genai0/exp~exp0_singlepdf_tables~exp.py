import os
import time
# multivector retriever
import openai
from dotenv import dotenv_values
config = dotenv_values(".env") 
openai.api_key  = config['OPENAI_API_KEY']

from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import DocArrayInMemorySearch, Chroma

import uuid
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough


# Ref
# intall tesseract!
# https://python.langchain.com/docs/integrations/providers/unstructured
# https://github.com/Unstructured-IO/unstructured

# Process PDF----------------------------------------------------------------------------
# See: https://unstructured-io.github.io/unstructured/core/partition.html#partition-pdf
start_time_partitionpdf = time.perf_counter()
raw_pdf_elements = partition_pdf(    
    filename= '/home/sebacastillo/genai0/bd/Apple_2023.pdf',
    # Unstructured first finds embedded image blocks
    extract_images_in_pdf=False,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path='bd/image',
)
end_time_partitionpdf = time.perf_counter()
duration_partition_pdf = end_time_partitionpdf - start_time_partitionpdf

# collect element by type
class Element(BaseModel):
    type: str
    text: Any

# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

# Tables
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))
# Text
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))


# Multivector Retriever-------------------------------------------------------------
# https://medium.com/@abatutin/is-the-langchain-multi-vector-retriever-worth-it-0c8565e8a8fd
# enhances the quality of RAG, particularly for table-intensive documents such as 10-K reports.

# Summary chain
prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# Apply to tables
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
# Apply to texts
texts = [i.text for i in text_elements]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})


# Save to Vector DB
# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings(),  persist_directory="bd/PDF_TextandTableSummaries_chroma_db")
store = Chroma(collection_name="parents_doc", embedding_function=OpenAIEmbeddings(),  persist_directory="bd/PDF_TextandTableParentsDoc_chroma_db")

# The storage layer for the parent documents
# store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))


# RAG-------------------------------------------------------------------------------
# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(temperature=0, model="gpt-4")

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("Create a summary of the 'CONSOLIDATED STATEMENTS OF OPERATION'")
chain.invoke("What are the Net Income of the company for the years 2021, 2022 and 2023")

# evaluation
# evaluation dataset
# https://medium.com/@abatutin/is-the-langchain-multi-vector-retriever-worth-it-0c8565e8a8fd
q_a_10_k = {
        "What is the value of cash and cash equivalents in 2022?": "16,253 $ millions",
        "What is the value of cash and cash equivalents in 2021?": "17,576 $ millions",
        "What is the net value of accounts receivable in 2022?": "2,952 $ millions",
        "What is the net value of accounts receivable in 2021?": "1,913 $ millions",
        "What is the total stockholders' equity? in 2022?": "44,704 $ millions",
        "What is the total stockholders' equity? in 2021?": "30,189 $ millions",
        "What are total operational expenses for research and development in 2022?": "3,075 $ millions",
        "What are total operational expenses for research and development in 2021?": "2,593 $ millions",
    }


