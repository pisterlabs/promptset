from langchain import PromptTemplate
from langchain.llms import OpenAI

template = """
  Compare the following two clauses and identify any inconsistencies between them. Do not identify inconsistencies that are immaterial or insubstantial (for example, ignore inconsistencies in language and style). Only identify material inconsistencies that alter the meaning of the text, including any differences in scope between the two clauses.

  Below are the two clauses:
  FIRST_CLAUSE: {first_clause}
  SECOND_CLAUSE: {second_clause}

  YOUR RESPONSE:
"""

prompt = PromptTemplate(
  input_variables=("first_clause", "second_clause"),
  template=template,
)

llm = OpenAI(temperature=0, model_name="gpt-4")

def compare_clauses(first_clause, second_clause):
    prompt_with_clauses = prompt.format(first_clause=first_clause, second_clause=second_clause)
    compared_clauses = llm(prompt_with_clauses)
    return compared_clauses
