import os
from langchain.llms import Petals
from langchain import PromptTemplate, LLMChain

from dotenv import load_dotenv
load_dotenv()


llm = Petals(model_name="meta-llama/Llama-2-70b-chat-hf")
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
llm_chain.run(question)



# from transformers import AutoTokenizer
# from petals import AutoDistributedModelForCausalLM

# model_name = "enoch/llama-65b-hf"
# # You can also use "meta-llama/Llama-2-70b-hf", "meta-llama/Llama-2-70b-chat-hf",
# # "bigscience/bloom", or "bigscience/bloomz"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
# # Embeddings & prompts are on your device, transformer blocks are distributed across the Internet

# inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
# outputs = model.generate(inputs, max_new_tokens=5)
# print(tokenizer.decode(outputs[0]))  # A cat sat on a mat...

