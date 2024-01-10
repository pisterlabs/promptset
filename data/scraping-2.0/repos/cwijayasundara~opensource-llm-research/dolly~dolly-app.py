import os

from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# Get your HuggingFace token from https://huggingface.co/settings/token
HUGGINGFACEHUB_API_TOKEN = ''
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_id = "databricks/dolly-v2-3b"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 200})

template = """You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers 
to the user's questions.

{question}
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
question = "what is the difference between nuclear fusion and nuclear fission"

llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))