from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.chains import LLMChain

load_dotenv()

llm = OpenAI(temperature=0.9)

prompt_text = open("document_prompt", "r").read()

prompt = PromptTemplate(
    input_variables=["text", "fragments"],
    template=prompt_text,
)

chain = LLMChain(llm=llm, prompt=prompt)


def getPrediction(text, fragments):
    return chain.run(text=text, fragments=fragments)
