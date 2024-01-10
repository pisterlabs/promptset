import openai
from Config import OPENAI_API_KEY, PROMPT

class AIService:
    def __init__(self):
        pass

    def getAIResponse(self, inputText):
        #print(f"Getting AI response on: {inputText}.")
        openai.api_key = OPENAI_API_KEY
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", "content": PROMPT
                },
                {
                    "role": "user", "content": inputText
                }
            ]
        )
        if response.choices:
            with open("output.txt", "w") as file:
                file.write(response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()
        else:
            return None

