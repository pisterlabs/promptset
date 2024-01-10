import openai


class ChatBot:

    def __init__(self):
        openai.api_key = "Type in your API Key Here"

    def get_response(self, user_input):
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo",
                prompt=user_input,
                max_tokens=1000,
                temperature=0.5
                ).choices[0].text
            return response
        except openai.error.RateLimitError:
            response = "Rate Limit Reached"
            return response
if __name__ == "__main__":
    chatbox = ChatBot()
    response = chatbox.get_response("Write a joke about cats")
    print(response)