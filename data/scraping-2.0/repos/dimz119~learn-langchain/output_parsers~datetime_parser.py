from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate

output_parser = DatetimeOutputParser()
# print(output_parser.get_format_instructions())
"""
Write a datetime string that matches the following pattern: 

"%Y-%m-%dT%H:%M:%S.%fZ". Examples: 
1429-09-21T16:58:05.456924Z, 
0617-11-10T20:38:46.065784Z, 
1427-01-23T04:40:20.388565Z
"""
template = """Answer the users question:

{question}

{format_instructions}"""
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)
chain = LLMChain(prompt=prompt, llm=OpenAI())
output = chain.run("around when was bitcoin founded?")
print(output_parser.parse(output))
"""
2009-01-03 18:15:05
"""