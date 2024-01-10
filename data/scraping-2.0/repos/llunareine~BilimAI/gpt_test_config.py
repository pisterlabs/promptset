import os
import json
from langchain.llms import OpenAI
from  dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def test_query(my_text: str) -> str:
    prompt = f"""
    Using the provided text:

        "{my_text}"

         you are an expert in creating tests. I am giving you a text for constructing questions and with an answer. 
         generate questions without asking unnecessary questions, just generate questions and return it as a list of 
         objects based on the given text, this is example of response format:

        [
        {{
        {{
              "question": "Sample question?",
              "options": {{
                    "A": "Option A",
                    "B": "Option B",
                    "C": "Option C",
                    "D": "Option D"
              }},
       }},

        {{
            "question": "Sample question 2?",
            "options": {{
                        "A": "Option A",
                        "B": "Option B",
                        "C": "Option C",
                        "D": "Option D"
                  }},
      }} 
      }}
      ]
    just send the list at json questions without any words and and completely write the list to the end.

    warning: The correct answer should always be on option A    
    """
    llm = OpenAI(model_name="gpt-3.5-turbo", n=10, temperature=0)
    completion = llm(prompt=prompt)
    try:
        list_of_dicts = json.loads(completion)
        return list_of_dicts
    except json.JSONDecodeError:
        print("Invalid JSON string")