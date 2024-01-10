import openai


class OpenAIQuestionAnswering:
    def __init__(self, api_key):
        openai.api_key = api_key

    def answer_question(self, question):
        # Call the OpenAI API to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "I am an artificial intelligence that analyzes accounting financial statements."},
                      {'role': 'user', 'content': question}],
            max_tokens=1024,
            n=1,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
        )
        # Get the response text from the API response
        response_text = response['choices'][0]['message']['content']

        return response_text
