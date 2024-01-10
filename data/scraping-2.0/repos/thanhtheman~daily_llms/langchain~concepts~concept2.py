from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
template = "I will travel to {city}, what should I do there? Please response in 2 short sentences."
prompt = PromptTemplate(input_variables=["city"], template=template)
final_prompt = prompt.format(city="New York")
response = model([HumanMessage(content=final_prompt)])
print(response)

#remember - predict only takes a string as an input.
response_predict = model.predict("I will travel to New York, what should I do there? Please response in 2 short sentences.")
print(response_predict)