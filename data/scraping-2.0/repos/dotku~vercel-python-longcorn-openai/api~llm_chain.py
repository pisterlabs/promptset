from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9, max_tokens=100)
prompt = PromptTemplate(
    input_variables=["brand"],
    template="Is the company {brand} a black listed company by the US customs?",
)

chain = LLMChain(llm=llm, prompt=prompt)

if __name__ == "__main__":
    print(chain.generate(brand="Huawei"))
    # "Is the company Huawei a black listed company by the US customs?
    # Yes, Huawei is a black listed company by the US customs."
