from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

model_id = "lmsys/fastchat-t5-3b-v1.0"
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text2text-generation",
    model_kwargs={"temperature": 0, "max_length": 10000},
)

template = """{question}"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = """{
  "name": "John Doe",
  "age": 30,
  "city": "New York",
  "isStudent": false,
  "grades": [85, 92, 78],
  "address": {
    "street": "123 Main St",
    "zipCode": "10001"
  }
}
Generate 5 Similarity Inputs like this.
"""

result = llm_chain(question)
print(result['question'])
print("")
print(result['text'])