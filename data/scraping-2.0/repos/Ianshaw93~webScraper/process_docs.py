import json
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from shared_funcs import text_splitter


with open('blogs.txt', 'r') as f:
    articles = json.load(f)

# meta data is all but text
# use text splitter to split each article? or document first?
keys = list(articles[0].keys())
meta_data_array = []
for idx, article in enumerate(articles):
    current_meta_array = []
    meta_data = {k: article[k] for k in keys if k != 'text'}
    meta_data['index'] = idx
    current_meta_array.append(meta_data)
    meta_data_array.append(meta_data)

docs = text_splitter.create_documents([f['text']for f in articles], metadatas=meta_data_array)
embeddings = OpenAIEmbeddings()


import pinecone

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

# can further indexes be created on free tier?
index_name = "kot-youtube"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=1536  
)
# The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)

query = "how to train the tibialis anterior without training experience?"
docs = docsearch.similarity_search(query)
# issue - whole document returned, not just the relevant section

# QUERY LLM
from langchain.prompts import PromptTemplate

template = """
    Use only the information in the context to answer the question.
    Context: {context}
    Question: {question}
    If the answer is not in the context, DO NOT MAKE UP AN ANSWER.

"""
prompt = PromptTemplate.from_template(
        template
    )

# TODO: include text of all relevant docs
# TODO: remove index from db and make docs smaller
all_context = ('').join([f.page_content for f in docs])
final_prompt = prompt.format(context=all_context ,question=query)

# TODO: send prompt to chatgpt
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-ada-001", openai_api_key=OPENAI_API_KEY)
answer = llm(final_prompt)
pass

# from langchain.llms import OpenAI
# from langchain.chains import LLMChain, ConversationalRetrievalChain

# template = (
#     "Combine the chat history and follow up question into "
#     "a standalone question. Chat History: {chat_history}"
#     "Follow up question: {question}"
# )
# prompt = PromptTemplate.from_template(template)
# llm = OpenAI()
# question_generator_chain = LLMChain(llm=llm, prompt=prompt)
# chain = ConversationalRetrievalChain(
#     combine_docs_chain=docs,
#     retriever=Pinecone.as_retriever(),
#     question_generator=question_generator_chain,
# )
pass
# gcp-starter