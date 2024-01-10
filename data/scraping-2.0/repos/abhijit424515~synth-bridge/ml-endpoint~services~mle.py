from openai import OpenAI
from services.common import openai_api_key

client = OpenAI(api_key=openai_api_key)

# Calculate the relevance of the input code to the project description
def calculate_relevance(project_description, code):
    output = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"You have {project_description} and {code}, analyze the code and tell if it is relevant to the project description or not. If it is not relevant then explain the error and if it is relevant then explain the logic of the code and the purpose of each. Give output in less then 30 words. And the output format should be like first tell is the code relevent or not and after that expain. If the code is not relevant to project description then no need to expain the code. ALso if you find similar worda in {project_description} and {code} consider that too, in this case don't tell directly if the code is relevant or not, first tell the similar words and then tell if the code is relevant or not. ALso the output format should be like Yes/No code is relevant and then expalination. ANd if there is error Bold that part."}
      ]
    )
    return { "response": output.choices[0].message.content }

