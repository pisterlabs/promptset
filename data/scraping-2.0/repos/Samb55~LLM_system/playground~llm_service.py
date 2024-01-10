from openai import OpenAI

import google.generativeai as genai
import os 

class GPTService:
    def __init__(self):
        self.openai = OpenAI()
        self.model = "gpt-3.5-turbo"

    def get_response(self, user_input):
        completion = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a personal assistant helping me to answer my questions."},
                {"role": "user", "content": user_input},
            ]
        )
        return completion.choices[0].message.content
    

class geminiService:
    def __init__(self):
        GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
    def get_response(self, user_input):
        response = self.model.generate_content(user_input)
        return response



if __name__ == "__main__":
    print(GPTService().get_response("What is the captial of USA?"))
    print(geminiService().get_response("What is the captial of USA?").text)

