from langchain.tools import BaseTool
from langchain.base_language import BaseLanguageModel
from .utils import search_texts, load_texts, Text, Doc
import openai
import os

class FindContext(BaseTool):
    name = "FindContextinDocs"
    description = """This tool finds context within the pdfs to add for QA. You can use it to answer questions from the user or get info if in doubt. """

    def _run(self, query:str):
      #texts = load_texts('xu2004_texts.pkl')
      try:
        texts2 = load_texts('/HVACAgent/geoinfo3.pkl')
      except Exception:
          try:
              texts2 = load_texts('geoinfo3.pkl')
          except Exception as e:
            #print current directory
            print(os.getcwd())
            print(os.listdir())
            print("geoinfo.pkl not found")
            raise e
      #results = search_texts(texts, query, n=3,pprint=False)
      try:
        results2 = search_texts(texts2, query, n=4,pprint=False)
      except Exception as e:
        try:
          openai.api_key = os.getenv("OPENAI_API_KEY")
          results2 = search_texts(texts2, query, n=2,pprint=False)
        except Exception as e:
          print(f"RetryError, posibly OPENAI_API_KEY not set currently OpenAI_API_KEY={os.getenv('OPENAI_API_KEY')}")
          raise e


  
      combined_list = results2
      sorted_list = sorted(combined_list, key=lambda item: item.similarity, reverse=True)
      final_list = [item.text for item in sorted_list]
      answer = ','.join(final_list)+', with this context a llm can try aswering the question'

      return answer

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")