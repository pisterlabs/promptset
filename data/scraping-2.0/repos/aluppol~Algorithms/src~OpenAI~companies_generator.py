import openai
import os
import json

openai.api_key = os.environ["OPENAI_KEY"]


def get_completion(question: str, model='gpt-3.5-turbo') -> str:
    messages = [{'role': 'user', 'content': question}]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,   # this is the degree of randomness of the model's output
        )
    except Exception as e:
        print('Error with OpenAI API: ', e)
        return ''
    return response.choices[0].message['content']


def get_list_of_companies_in_industry(industry_description: str) -> list:
    """
    Returns a list of companies in the given industry
    """
    message = f"""
    You will be provided a specific description of the industry delimited by ```.
    You task is to provide result in JSON format by following the next steps:
    1. Summarise all data provided in the industry description and determine the key points of it.
    2. Find companies that will fit key point of the industry description from step 1.
    3. Evaluate how good each company fit the key points of the industry description from step 1.
    4. Evaluate how popular and large each company on industry market are.
    5. Range all companies that you have from companies with higher score from step 3 and 4 to companies with lower score.
    6. Take top 10 companies from the list you have from step 5.
    7. Provide the answer in JSON format with the following structure:
        \"[
            {{
                "name": <company name>,
                "description": <company description>,
                "score": <company score>
            }}
        ]\"
    8. Double check that in the response you include only data in JSON format, and there is no other data.
    ```
        {industry_description}
    ```
    """
    response = get_completion(message)
    try:
        result = json.loads(response)
    except Exception as e:
        print('Error with response format: ', e)
        print(response)
        return []
    return result

print(get_list_of_companies_in_industry('Travel industry. Companies that provide housing: hotels, rooms, flats, houses, etc.'))
