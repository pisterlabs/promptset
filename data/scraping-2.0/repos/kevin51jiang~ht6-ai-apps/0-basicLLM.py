import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain


load_dotenv(find_dotenv())

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command-nightly")
llm_chain = LLMChain(prompt=prompt, llm=llm)


question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

answer = llm_chain.run(question)
print("Answer: ", answer)

# # Embeddings
# from langchain.embeddings import CohereEmbeddings

# embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

# query_result = embeddings.embed_query("Hello")
# print("query result: ", query_result)

# doc_result = embeddings.embed_documents(["Hello there", "Goodbye"])
# print("Doc result: ", doc_result)



