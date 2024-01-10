import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict
import re

load_dotenv()

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)

# Finding and extracting the JSON content; ignores all string that comes before and after the ```json <JSON> ``` marker
def clean_response(response: str):
  pattern = r"```json\n(.*?)\n```"

  match = re.search(pattern, response, flags=re.DOTALL)

  if match:
      cleaned_response = match.group(1)
      return cleaned_response
  else:
      raise Exception("No JSON content found in response")

def gpt_call(sys_msg: Optional[str], prompt: str, json_format: Dict) -> str:
  format_specification = f"""
I will ask you questions and you will respond. Your response should be in JSON format, with the following structure.:
```json
{json_format.strip()}
```
"""

  print(format_specification + prompt)

  messages = []

  if sys_msg is not None:
    messages.append({"role": "system", "content": sys_msg})

  messages.append({"role": "user", "content": format_specification + prompt})

  completion = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=messages
  )

  print(completion.choices[0].message.content)

  return completion.choices[0].message.content

# Test Cases
if __name__ == "__main__":
  from pydantic import BaseModel

  class Response(BaseModel):
    score: int

  json_format = """
{
  "product": str
}
"""
  sys_msg = None
  business_problem = """
The construction industry is indubitably one of the significant contributors to global waste, contributing approximately 1.3 billion tons of waste annually, exerting significant pressure on our landfills and natural resources. Traditional construction methods entail single-use designs that require frequent demolitions, leading to resource depletion and wastage. 
"""
  business_solution = """
Herein, we propose an innovative approach to mitigate this problem: Modular Construction. This method embraces recycling and reuse, taking a significant stride towards a circular economy.   Modular construction involves utilizing engineered components in a manufacturing facility that are later assembled on-site. These components are designed for easy disassembling, enabling them to be reused in diverse projects, thus significantly reducing waste and conserving resources.  Not only does this method decrease construction waste by up to 90%, but it also decreases construction time by 30-50%, optimizing both environmental and financial efficiency. This reduction in time corresponds to substantial financial savings for businesses. Moreover, the modular approach allows greater flexibility, adapting to changing needs over time.  We believe, by adopting modular construction, the industry can transit from a 'take, make and dispose' model to a more sustainable 'reduce, reuse, and recycle' model, driving the industry towards a more circular and sustainable future. The feasibility of this concept is already being proven in markets around the globe, indicating its potential for scalability and real-world application.
"""
  prompt = f"""
Given the following business idea and solution, score how adequately the solution solves the problem on a scale of 1-10. Do not output anything else.

Business Problem:

{business_problem.strip()}

Business Solution:

{business_solution.strip()}
"""
  response = gpt_call(sys_msg, prompt, json_format)

  cleaned_response = clean_response(response)

  print(cleaned_response)
  
  validated_response = Response.model_validate_json(cleaned_response)

  print(validated_response)