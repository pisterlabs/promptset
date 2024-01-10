from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
# chain to take in user's location -> suggesting a dish, the given dish's ingredients, and the recipe
template1= "Given the location, {location} please suggest a classic dish"
prompt1 = PromptTemplate(input_variables=["location"], template=template1)

dish_chain = LLMChain(llm=model, prompt=prompt1)

template2= "Given the name of the dish {dish}, please list the ingredients and the recipe"
prompt2 = PromptTemplate(input_variables=["dish"], template=template2)
recipe_chain = LLMChain(llm=model, prompt=prompt2)

chain = SimpleSequentialChain(chains=[dish_chain, recipe_chain], verbose=True)
chain.run("Rome")