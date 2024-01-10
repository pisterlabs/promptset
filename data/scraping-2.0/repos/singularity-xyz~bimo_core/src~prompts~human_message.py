from langchain.prompts import HumanMessagePromptTemplate

default_human_template = "{human_message}"
default_human_message_prompt = HumanMessagePromptTemplate.from_template(default_human_template)