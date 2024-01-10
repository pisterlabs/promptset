import openai
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API")
openai.api_key = api_key

def get_recommendation(experiences: list[str], city: str, additional: str|None) -> str:
    experience_list = [ "- " + experience for experience in experiences]
    experience_list_text = "\n".join(experience_list)
    prompt = f"""Following is a list of restaurant preferences of a person. Based on this list, suggest some other restaurants in the city {city}(United States) to this person:\n{experience_list_text}
    """
    if additional:
        prompt = f"{prompt}\n Also, keep the following in mind for each suggestion: \n```\n{additional}\n```"
    print("PROMPT:") 
    print(prompt)

    result = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Assume you are an expert at recommending restaurants to people based on their preferences."},
            {"role": "user", "content": prompt},
        ]
    )
    print("RESPONSE: ")
    print(result.choices[0].message.content)
    return result.choices[0].message.content
