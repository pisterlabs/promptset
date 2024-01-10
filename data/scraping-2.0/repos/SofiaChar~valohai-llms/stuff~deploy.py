from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate, LLMChain


model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)
local_llm = HuggingFacePipeline(pipeline=pipe)

template = """Question: {question}
Answer: Let's think step by step.
Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=local_llm)

question1 = "What is the capital of Germany?"
print(llm_chain.run(question1))

from langchain.chains import VectorDBQA

qa = VectorDBQA.from_chain_type(llm=local_llm, chain_type="stuff", vectorstore=vectordb)

question = "What is amazon sagemaker?"

llm_output = llm_chain.run(question)
qa_output = qa.run(question)

# LLM without the vector DB
print("LLM Output: ", llm_output)

# LLM with the vector DB
print("Vector DB Output: ", qa_output)