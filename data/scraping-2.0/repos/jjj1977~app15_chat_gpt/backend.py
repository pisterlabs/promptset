import openai


class Chatbot:
    def __init__(self):
        openai.api_key = "sk-P3KsLf7gnCLUUtCtmZeYT3BlbkFJkSX64PBzELtT9ssDDBp"

    def get_response(self, user_input):
        response = openai.Completion.create(engine="text-davinci-003",prompt=user_input,
                                            max_tokens=3000, temperature=0.5).choice