import openai

openai.api_key = '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'

class OpenAIChatBot:
    def init(self):
        self.prompt = "The following is a conversation with a chatbot:\nYou: "

    def get_response(self, user_input):
        self.prompt += user_input + "\nChatbot: "
        response = openai.Completion.create(
          model="text-davinci-003", # You can change this to other available models
          prompt=self.prompt,
          max_tokens=10
        )
        answer = response.choices[0].text.strip()
        self.prompt += answer + "\nYou: "
        return answer

    def chat(self):
        print("OpenAIChatBot: Hello! Type 'bye' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['bye', 'goodbye']:
                print("OpenAIChatBot: Goodbye! Talk to you later.")
                break

            response = self.get_response(user_input)
            print(f"OpenAIChatBot: {response}")


if name == 'main':
    bot = OpenAIChatBot()
    bot.chat()