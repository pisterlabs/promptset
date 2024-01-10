import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

open_api_key = "<open-ai key>"

os.environ['OPENAI_API_KEY'] = open_api_key

pinecone.init(
    api_key="<pincone-key>",
    environment="<pincone env>",
)

llm = OpenAI(openai_api_key=open_api_key,
             temperature=0.5)
embedding = OpenAIEmbeddings(openai_api_key=open_api_key)

summary_chain = load_summarize_chain(llm, chain_type="refine")

summarize_document_chain = AnalyzeDocumentChain(
    combine_docs_chain=summary_chain)

docsearch = Pinecone.from_existing_index("chemistry-2e", embedding=embedding)
docs = docsearch.similarity_search("what is Enthalpy?", k=10)
inputs = " ".join([doc.page_content for doc in docs])
print("---------")
print(inputs)
print("---------")


print(summarize_document_chain.run(inputs))
