from getpass import getpass
import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

#setup tokens
HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

print(f"HuggingFaceHub API token set successfully {HUGGINGFACEHUB_API_TOKEN}")

repo_id = "stabilityai/stablelm-tuned-alpha-3b"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 500})

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Who won the FIFA World Cup in the year 1994? "
print(llm_chain.run(question))