from langchain.prompts import PromptTemplate

summary_template = """Give the following law extract, translate it to {language}, remove any lines that doesn't make sense, and generate a long summary which highligts the most important parts of the law such as the title of the law, current situation, analysis, goals, conclusions, and comission's recommendations:
{text}


OUTPUT:
"""

MAP_PROMPT = PromptTemplate(input_variables=["text", "language"], template=summary_template)
REDUCE_PROMPT = PromptTemplate(input_variables=["text", "language"], template=summary_template)