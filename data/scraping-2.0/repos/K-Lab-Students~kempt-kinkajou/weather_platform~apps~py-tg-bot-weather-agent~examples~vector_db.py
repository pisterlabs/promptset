from langchain.llms import OpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import faiss
import numpy as np


class LocalVectorDB:
  def __init__(self):
    self.index = None
    self.vector_dim = None

  def index_vectors(self, vectors):
    self.vector_dim = vectors.shape[1]
    self.index = faiss.IndexFlatL2(self.vector_dim)
    self.index.add(vectors)

  def search(self, query_vector, k):
    distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
    return distances[0], indices[0]

OPENAI_API = "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"

template = """
You are an expert data scientist with an expertise in building deep learning models.
Explain the concept of {concept} in a couple of lines
"""
prompt = PromptTemplate(
  input_variables=['concept'],
  template=template,
)

llm = OpenAI(model_name="text-davinci-003", openai_api_key=OPENAI_API)
# msg = llm("explain large language models in one sentance")
# msg = llm(prompt.format(concept="regularization"))
# print(msg)

chain = LLMChain(llm=llm, prompt=prompt)
print("First Chain: ", chain.run("autoencoder"))

second_prompt = PromptTemplate(
  input_variables=["ml_concept"],
  template="Turn the concept description of {ml_concept} and explain it to me like I'm five",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)
explanation = overall_chain.run("autoencoder")
print("Second Chain: ", explanation)

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=100,
  chunk_overlap=0,
)
texts = text_splitter.create_documents([explanation])
print("Page content: ", texts[0].page_content)

embeddings = OpenAIEmbeddings(model_name="ada", openai_api_key=OPENAI_API)

query_result = embeddings.embed_query(texts[0].page_content)
print("Vector Query Result: ", query_result)


# all_embeddings = []
# for text in texts:
#     embedding = embeddings.embed_query(text.page_content)
#     all_embeddings.append(embedding)
#
# all_embeddings = np.array(all_embeddings)

all_embeddings = []
all_embeddings.append(query_result)
all_embeddings = np.array(all_embeddings)

local_db = LocalVectorDB()
local_db.index_vectors(all_embeddings)
distances, indices = local_db.search(query_result, 5)
print(print(embeddings[indices]))
print(distances)



