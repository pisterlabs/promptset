import time
from dotenv import load_dotenv, find_dotenv
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoModelForSeq2SeqLM,
)

_ = load_dotenv(find_dotenv())  # read local .env file


start = time.time()


# ---------------------------
# Text Completion in a Chain with Prompt Template
# ---------------------------

model_id = "facebook/blenderbot-1B-distill"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation", model=model, tokenizer=tokenizer, max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(
    llm=local_llm,
    prompt=prompt,
)

result = chain.run("colorful socks")
print(result)

result = chain.run("jet engine cars")
print(result)


end = time.time()
print(f"NN takes: {end - start} sec.")
