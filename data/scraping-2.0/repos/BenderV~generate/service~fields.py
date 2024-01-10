from typing import List

import yaml
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

PREDICT_COLUMNS_TEMPLATE = """
Try to predict the columns for each list
Type and description are optional
Type options are string, integer, float, boolean

FORMAT YAML
- name
  type
  description

### 

title: "belgium cities (top 10)"
fields:
  - name: "name"
    type: "string"
    description: "city name"
  - name: "population"
    type: "integer"
    description: "in millions of habitant"

---

title: "best surfer in the world"
fields:
   - name: "name"
     type: "string"
   - name: "country"
     type: "string"
   - name: "age"
     type: "integer"

title: "{{ title }}
fields:    
"""

llm = OpenAI(temperature=0.9, stop=["---"])


def guess_fields(title: str) -> List[dict]:
    # Predict columns
    prompt = PromptTemplate(
        template_format="jinja2",
        template=PREDICT_COLUMNS_TEMPLATE,
        input_variables=["title"],
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_key="fields")
    output = chain({"title": title})
    # Should probably check if output is valid
    fields = yaml.safe_load(output["fields"])
    print("fields", fields)
    return fields
