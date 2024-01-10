from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser

import settings

OPEN_AI_API_KEY = settings.OPEN_AI_API_KEY

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

def main():
  chat_model = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, model_name="gpt-3.5-turbo")

  template = """You are a helpful assistant who generates comma separated lists.
  A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
  ONLY return a comma separated list, and nothing more."""
  human_template = "{text}"

  chat_prompt = ChatPromptTemplate.from_messages([
      ("system", template),
      ("human", human_template),
  ])

  chain = chat_prompt | chat_model | CommaSeparatedListOutputParser()
  output = chain.invoke({"text": "colors"})

  print(output) 


if __name__ == "__main__":
    main()