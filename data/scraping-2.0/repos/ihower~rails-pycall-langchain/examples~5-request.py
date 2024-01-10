# https://python.langchain.com/en/latest/modules/chains/examples/llm_requests.html


from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate




template = """Between >>> and <<< are the raw search result text from google.
Extract the answer to the question '{query}' or say "not found" if the information is not contained.
Use the format
Extracted:<answer or "not found">
>>> {requests_result} <<<
Extracted:"""



PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)

llm = OpenAI(temperature=0, openai_api_key= "sk-your-key-here")

chain = LLMRequestsChain(llm_chain = LLMChain(llm=llm, prompt=PROMPT) )

question = "What are the Three (3) biggest countries, and their respective sizes?"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+")
}

print(chain.run(inputs))
