import json
import requests
from iteration_utilities import deepflatten
from typing import List, Optional, Type, Union
from pydantic import BaseModel
from langchain.tools import BaseTool


class BlawxCategory(BaseModel):
    """
    The data structure for a fact about membership in a category.
    """

    from_ontology: bool
    type: str
    category: str
    object: str

class BlawxAttributeBinary(BaseModel):
    """
    The data structure for a binary attribute.
    """

    from_ontology: bool
    type: str
    attribute: str
    object: str
    value: str

class BlawxAttributeUnary(BaseModel):
    """
    The data structure for a unary attribute.
    """

    from_ontology: bool
    type: str
    attribute: str
    object: str

class BlawxRelationship(BaseModel):
    """
    The data structure for a relationship fact.
    """

    from_ontology: bool
    type: str
    relationship: str
    parameter1: str
    parameter2: str
    parameter3: str
    # This would need to be extended later.

BlawxFact = Union[BlawxCategory, BlawxRelationship, BlawxAttributeBinary, BlawxAttributeUnary]

class BlawxFacts(BaseModel):
    """
    This documents the data structure used by the interview and run endpoints.
    """
    facts: List[BlawxFact]

class PAInterview(BaseTool):
    name = "permitted_uses"
    description = """
Useful for finding out whether a use of information is permitted under the Privacy Act.

Requires you to know the ontology first.

The "type" of each fact should be the string "true", and the "from_ontology" value should be set to boolean false.

Pieces of information, individuals, government institutions and purposes must be defined in category facts before they are used in attribute and relationship facts.

All text values should be strings that start with lowercase letters and do not contain spaces.

The facts must exclusively use the category, attribute, and relationship predicates set out in the ontology.
Values should not be provided for boolean attributes.

If an entity is described as a government entity, include it in both the "entity" and "government_institution"
categories.

If there is an information, also specify in the facts that it is recorded.

Encode only facts that were obtained from the user. Do not encode facts derived from previous answers.
"""

    def _run(self, facts):
        return privacy_interview({"facts": facts})

    def _arun(self, input):
        raise NotImplementedError("The permitted_uses tool does not support asynchronous requests.")

    args_schema: Optional[Type[BaseModel]] = BlawxFacts

def privacy_interview(input):
  response = requests.post('https://dev.blawx.com/jason/privacy-act/test/permitted_use/run/',json=input)
  #print(response.text)
  package = json.loads(response.text)
  if len(package['Answers']):
    explanation = ''.join(list(deepflatten(package['Answers'][0]['Models'][0]['Tree'])))
    gch = explanation.find('The global constraints hold')
    if gch:
       explanation = explanation[:gch]
    return "I should use only the following information to answer the question: " + package['Answers'][0]['Variables']['Info'] + " can be used by " + package['Answers'][0]['Variables']['Inst'] + " for the purpose of " + package['Answers'][0]['Variables']['Purpose'] + " because "  + explanation
  else:
    return "I should use only the following information to answer the question: There is no evidence based on these facts to conclude that there is any permitted use of information under the AITA."