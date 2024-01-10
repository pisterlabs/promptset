import openai


class Chatbot():
    def __init__(self):
        super().__init__()
        openai.api_key = "OPEN_API_KEY"

    def get_response(self, user_input):
        response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=user_input,
                    max_tokens=1000,
                    temperature=0.5
        ).choices[0].text
        # temperature can vary between 0 and 1
        # towards 0 -> gives more accurate and less diverse answer
        # towards 1 -> gives less accurate and more diverse answer
        return response

# if __name__ == '__main__':
#     chatbot = Chatbot()
#     print(chatbot.get_response('Hello.'))
