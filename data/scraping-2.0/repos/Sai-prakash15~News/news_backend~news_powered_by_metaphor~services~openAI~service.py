import openai

from news_powered_by_metaphor.config import OPENAI_API_KEY



class OpenAIService:
    def __init__(self, use_prompt=0):
        openai.api_key = OPENAI_API_KEY

    def request_description_from_content(self, html_content):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an efficient assistant that quickly processes HTML content and provides a concise 2-3 line summary in less than half a second."},
                {"role": "user", "content": html_content},
            ],
        )

        openai_result = completion.choices[0].message.content

        print("==========openai response:==========\n", openai_result)

        return openai_result