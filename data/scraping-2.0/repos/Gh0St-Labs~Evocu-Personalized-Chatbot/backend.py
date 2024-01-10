import openai

class TheChatBot():
    openai.api_key = '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'

    def get_response(self, user_input):
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=user_input,
            max_tokens=4000,
            temperature=0.5
        ).choices[0].text
        return response


if __name__ == '__main__':
    chatbot = TheChatBot()
    response = chatbot.get_response(user_input="Tell me a joke on harsh and toxic people")
    print(response)