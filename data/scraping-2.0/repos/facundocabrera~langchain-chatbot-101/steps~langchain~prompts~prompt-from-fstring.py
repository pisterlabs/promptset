# Define a fstring and use it as a template for a prompt
from langchain.prompts import PromptTemplate

fstring_template = """Tell me a {adjective} joke about {content}"""
prompt = PromptTemplate.from_template(fstring_template)

print(prompt.format(adjective="funny", content="chickens"))