from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(openai_api_key=" ", temperature=0.2)

prompt = PromptTemplate(
    input_variables=["customer_name", "email", "agent_name"],
    template="""
       You are an AI assistant that is optimized to write concise, friendly and professional emails to customers.
       Write an email to customer with a name {customer_name}, that contains following optimized content : {email}
       Email signature should contains the name of customer agent : {agent_name}
       
    
       """,
)
name = input("Enter customer name")
email = input("What is the response you want to give")
agent_name = input("Enter Agent Name")

response = llm(prompt.format(customer_name=name, email=email, agent_name=agent_name))
print(response)
