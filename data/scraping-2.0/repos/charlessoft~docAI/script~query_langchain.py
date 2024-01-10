from langchain import PromptTemplate, LLMChain
from langchain.llms import AzureOpenAI

from config import *

# llm = NewAzureOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
# llm = AzureOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)
#
# # llm.deployment_name = 'ChatGPT-0301'
# llm.deployment_name = 'code-davinci-002'
#
# template = """You are a teacher in physics for High School student. Given the text of question, it is your job to write a answer that question with example.
# Question: {text}
# Answer:
# """
# prompt_template = PromptTemplate(input_variables=["text"], template=template)
# answer_chain = LLMChain(llm=llm, prompt=prompt_template)
# answer = answer_chain.run("What is the formula for Gravitational Potential Energy (GPE)?")
# print(answer)
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

embeddings = OpenAIEmbeddings(document_model_name="text-embedding-ada-002", chunk_size=1)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV # next to api key in console
)


docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)


# restaurant_template = """
# 我想让你成为一个给新开餐馆命名的顾问。
# 给我返回一个餐馆名字的名单. 每个餐馆名字要简单, 朗朗上口且容易记住. 它应该和你命名的餐馆类型有关.
# 关于{restaurant_desription} 这家餐馆好听名字有哪些?
# """

# restaurant_template = """
# You are an AI assistant, you will help write the code,You just need to give the code snippet"
# 请帮忙编写{restaurant_desription}代码
# """

# input_variables=["text_input"],
restaurant_template="You are an AI assistant, you will help write the code,You just need to give the code snippet:\n\n {restaurant_desription}"


#创建一个prompt模板
prompt_template=PromptTemplate(
    input_variables=["restaurant_desription"],
    template=restaurant_template
)



llm = AzureOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
llm.deployment_name = 'code-davinci-002'

chain = load_qa_chain(llm, chain_type="stuff")

# query = "How to generate a barcode bitmap from a string?"
query = '如何从字符串生成条形码位图'
# query = "今天天气如何?"
docs = docsearch.similarity_search(query,
                                   include_metadata=True, namespace=namespace)

answer = chain.run(input_documents=docs, question=query)
print(answer)
