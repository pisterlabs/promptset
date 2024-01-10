from langchain.chains import LLMChain
from langchain import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
  input_variables=["company","product"],
  template="What is a good name for {company} that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# for multiple variables we can input them all using a dictionary.

print(chain.run({
  'company': "ABC Startup",
  'product': "colorful socks"
}))


