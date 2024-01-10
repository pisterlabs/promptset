import json
from langchain.schema import BaseOutputParser

class JsonOutputParser(BaseOutputParser):
  def parse(self, text: str) -> list:
    try:
      return json.loads(text)
    except:
      return []