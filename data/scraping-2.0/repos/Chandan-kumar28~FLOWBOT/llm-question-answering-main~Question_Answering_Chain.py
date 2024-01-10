# Import the INSTRUCTOR embedding from an external module called InstructorEmbedding
from InstructorEmbedding import INSTRUCTOR

# Import necessary modules and classes from Langchain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Milvus  # Import Milvus for vector storage
from langchain.embeddings import HuggingFaceInstructEmbeddings  # Import embedding function

# Loading Milvus vector database for querying
vector_db: Milvus = Milvus(
    embedding_function=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base"),
    collection_name="SunbirdData",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

# Build a template for the question-answering prompt
template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

# Initialize a language model using the Hugging Face model repository
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature": 0.5, "max_length": 512},
    huggingfacehub_api_token='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'
)

# Create a prompt template for the QA chain using the defined template
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Initialize a RetrievalQA chain with the language model, retriever, and prompt template
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question="What is sunbird?"
ans= qa_chain({"query": question})
print(ans)
