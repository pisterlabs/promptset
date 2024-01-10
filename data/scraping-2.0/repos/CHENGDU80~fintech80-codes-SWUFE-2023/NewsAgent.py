import openai as openai


class NewsAgent:
    def __init__(self, api_key, model, messages=[]):
        openai.api_key = api_key
        self.model = model
        self.messages = messages

        # self.browser = webdriver.Chrome()

    def run(self):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=0
        )
        return response.choices[0].message['content']

    def environmental_feedback(self, feedback):
        self.messages.append({
            'role': 'system',
            'content': feedback
        })

    def continue_conversation(self, user_input):
        self.messages.append({
            'role': 'user',
            'content': user_input
        })

        assistant_response = self.run()
        self.messages.append({
            'role': 'assistant',
            'content': assistant_response
        })

        return assistant_response