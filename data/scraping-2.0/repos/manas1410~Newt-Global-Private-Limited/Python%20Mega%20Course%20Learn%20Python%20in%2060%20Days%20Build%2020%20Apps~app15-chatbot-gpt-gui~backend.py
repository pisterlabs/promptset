from openai import OpenAI

client = OpenAI(api_key="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")


class Chatbot:
    def get_response(self, user_input):
        response = client.completions.create(model="text-davinci-003",
                                             prompt=user_input).choices[0].text
        return response


if __name__ == "__main__":
    chatbot = Chatbot()
    response = chatbot.get_response("Write a joke about animals.")
    print(response)
