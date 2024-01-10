from langchain import PromptTemplate
from langchain.llms import OpenAI

template = """
  You are a lawyer drafting some amendments.  Propose some amendments to the following clause which are favourable to the party indicated below

  Below are the clause, and the favorable party:
  CLAUSE: {clause}
  FAVORABLE_PARTY: {favorable_party}

  YOUR RESPONSE:
"""

prompt = PromptTemplate(
  input_variables=("clause", "favorable_party"),
  template=template,
)

llm = OpenAI(temperature=0, model_name="gpt-4")

def suggest_amendments(clause, favorableParty):
    prompt_with_clauses = prompt.format(clause=clause, favorable_party=favorableParty)
    compared_clauses = llm(prompt_with_clauses)
    return compared_clauses
