#!pip install langchain llama-cpp-python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.llms import LlamaCpp

### Cloud
# model = OpenAI()

### Edge
# model = LlamaCpp(model_path="./models/llama-7b-ggml-v2-q4_0.bin", verbose=True, n_threads=8, n_gpu_layers=26)
# model = LlamaCpp(model_path="./models/stable-vicuna-13B-ggml_q4_0.bin", verbose=True, n_threads=8, n_gpu_layers=10)
model = LlamaCpp(model_path="./models/koala-7B.ggml.q4_0.bin", verbose=True, n_threads=8, n_gpu_layers=26)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=model)
question = "What is the primary colour, that is associated with starsign Aries?"

response=llm_chain.run(question)
print("Response: " + response)

