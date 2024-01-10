'''Learning to use chains'''
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain

llmOpenAI = ChatOpenAI(openai_api_key="TU_API_KEY")

prompt1 = ChatPromptTemplate.from_template(
    "Sugiereme un nombre para una empresa \
     que produce {product}? ubicada en {location}")
chain1 = LLMChain(llm=llmOpenAI, prompt=prompt1, output_key="company_name")

prompt2 = ChatPromptTemplate.from_template(
    "Escribe un tagline para una empresa llamada {company_name}")
chain2 = LLMChain(llm=llmOpenAI, prompt=prompt2, output_key="tagline")

prompt3 = ChatPromptTemplate.from_template(
    "Escribe un ad para poner en Google Ads para la empresa {company_name},\
          y usa el tagline de la empresa, que es {tagline}")
chain3 = LLMChain(llm=llmOpenAI, prompt=prompt3, output_key="ad")

sequential_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["product", "location"],
    output_variables=["company_name", "tagline", "ad"],
    verbose=True)

ad = sequential_chain({"product": "aceite de oliva", "location": "Valencia"})
print(ad)
