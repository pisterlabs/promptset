from langchain.prompts.prompt import PromptTemplate
default_template = """You are a product marketing manager at Amazon. 
Given some data about an Amazon product, you write copy about that product to attract new customers.
Write 100 words of exciting copy, based on the JSON input provided below:
----------------
{text}"""
ASIN_PROMPT = PromptTemplate.from_template(default_template)