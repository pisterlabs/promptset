# %%

import os

from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# %%
folder_path = "data/md_marker"
md_list = []
for file in os.listdir(folder_path):
    if file.endswith(".md"):
        md_list.append(file)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = []
for path in md_list:
    loader = TextLoader(f"{folder_path}/{path}")
    data = loader.load()
    print(f"Loaded {path}")
    splits = text_splitter.split_documents(data)
    print(f"there are {len(splits)} chunks in {path}")
    docs.extend(splits)
# %%
sentence_t_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=sentence_t_emb)
vectorstore.add_documents(documents=docs)  # it took 15 minutes

# %%


template = """
You are a respected spiritual teacher, Rob Burbea.
Try to distill the following pieces of context to answer the question at the end.
Question is asked by a student.
If you don't know the answer, just say that you don't know.
Don't try to make up an answer.
Use five sentences maximum and keep the answer as concise as possible.
Avoid answering questions that are not related to the dharma.

{context}
Question: {question}
Helpful Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

retriever = vectorstore.as_retriever(search_kwargs={"k": 7})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
repo_id = "SkunkworksAI/tinyfrank-1.4B"

# HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 256}
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "What is attention?"
a = rag_chain.invoke(question)
print(a)
# %%
question = "What movies should I watch?"
a = rag_chain.invoke(question)
print(a)
# %%
