import os
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # for using HugginFace models
from langchain import HuggingFaceHub, HuggingFacePipeline, PromptTemplate, LLMChain
from langchain.chains import RetrievalQA

load_dotenv("../.env")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_name = os.getenv("LAMINI_783M_PATH")  # LAMINI_1_5B_PATH
persist_directory = os.getenv("CHROMA_DB")

# Embeddings
embeddings = HuggingFaceEmbeddings()

# Initialize PeristedChromaDB
chroma = Chroma(
    collection_name="corporate_db",
    embedding_function=embeddings,
    persist_directory=persist_directory,
)

question = "Why Milvus Vector Database is used in corporate brain?"

llm = HuggingFaceHub(
    repo_id=model_name,
    model_kwargs={"temperature": 0, "max_length": 512},
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chroma.as_retriever(),
    input_key="question",
)

responses = chain.run(question)


# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_name,
#     task="text-generation",
#     # model_kwargs={"temperature": 0.5, "max_length": 512},
# )


# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = PromptTemplate(template=template, input_variables=["question"])

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# responses = llm_chain.run(question)


# print(len(responses))
print(responses)
