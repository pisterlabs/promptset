import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":1e-10})

question = "When was Google founded?"

llm_chain = LLMChain(llm=llm, prompt=template, input_variables=["question"])
answer = llm_chain(question=question)
print(answer)

