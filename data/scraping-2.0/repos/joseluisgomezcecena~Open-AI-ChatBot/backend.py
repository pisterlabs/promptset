import openai
#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
class Chatbot:
    def __init__(self):
        openai.api_key = "YOUR_API_KEY"

    def get_response(self, user_input):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt= user_input,
            max_tokens=2500,
            temperature=0.5,
        ).choices[0].text
        return response

if __name__ == "__main__":
    chatbot = Chatbot()
    response = chatbot.get_response(user_input="I am feeling sad")
    print(response)
    # print(chatbot.get_response("I am feeling sad"))
