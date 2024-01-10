from langchain.prompts import PromptTemplate

llm = OpenAI(openai_api_key="sk-...")

template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables=["fruit"])

response = llm(template.format(fruit="apple"))

print(response)