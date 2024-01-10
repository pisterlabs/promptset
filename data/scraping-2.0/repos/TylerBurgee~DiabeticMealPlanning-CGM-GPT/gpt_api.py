"""
Author: Tyler J. Burgee
"""

# IMPORT MODULES
import openai

class GPT_API:

    def __init__(self, api_key: str, org_id: str, model='gpt-3.5-turbo') -> None:
        """Defines the constructor for a GPT_API object"""
        self.api_key = api_key
        self.org_id = org_id
        self.model = model

        openai.api_key = self.api_key
        openai.organization = org_id

    def send_prompt(self, prompt: str):
        """Sends a prompt to the GPT API"""
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

if __name__ == '__main__':
    api_key = 'API_KEY_HERE'
    org_id = 'ORGANIZATION_ID_HERE'

    # INSTANTIATE GPT_API OBJECT
    api = GPT_API(api_key, org_id)

    prompt = 'Hello, ChatGPT!'
    response = api.send_prompt(prompt)

    print(response)
