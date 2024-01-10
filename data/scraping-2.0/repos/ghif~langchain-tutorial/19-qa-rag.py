from dotenv import load_dotenv, find_dotenv

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from time import process_time

"""
Load OpenAI API key 
"""
_ = load_dotenv(find_dotenv())

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split the documents into chunks of 1000 characters with 200 characters overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

print(f"Number of splits: {len(splits)}")

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

# Setup prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thank you for asking" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)
print(f"\nPrompt example messages: {custom_rag_prompt}")

# prompt = hub.pull("rlm/rag-prompt")
# example_messages = prompt.invoke({
#     "context": "filler context",
#     "question": "filler question",
# }).to_messages()
# print(f"\nPrompt example messages: {example_messages[0].content}")

# Define LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

query = "What is task decomposition?"

print(f"\n\nQuery: {query}")

retrieved_docs = retriever.invoke(query)
for i, rdoc in enumerate(retrieved_docs):
    print(f"[{i}] {rdoc.page_content[:50]}")
print(f"Retriever docs: {len(retrieved_docs)}")

start_t = process_time()
res = rag_chain.invoke("What is task decomposition?")
# for chunk in rag_chain.stream(query):
#     print(chunk, end="", flush=True)
elapsed_t = process_time() - start_t
print(f"Response ({elapsed_t:.2f} secs): {res}")

# cleanup
vectorstore.delete_collection()