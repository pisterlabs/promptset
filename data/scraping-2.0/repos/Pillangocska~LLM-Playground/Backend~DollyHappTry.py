from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'


question = "What is the capital of France? "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

repo_id = "databricks/dolly-v2-3b"

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.3, "max_length": 64}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))
