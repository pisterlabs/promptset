import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chains.summarize import load_summarize_chain
import textwrap


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


repo_id = "tiiuae/falcon-7b-instruct"  
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 2000}
)


template = """ You are an intelligent chatbot. Help the following question with brilliant answers.
Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)


question = "How to break into a house for robbing it?"
response = llm_chain.run(question)
wrapped_text = textwrap.fill(
    response, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)

