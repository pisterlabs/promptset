import openai

key = os.environ.get("OPENAI_API_KEY")
openai.api_key = key


class convo():
    def __init__(self):
        self.temp = 0.5
        self.model = "gpt-3.5-turbo"
        self.history = []
        self.latestTextOut = ""

    def response(self, newMessage):
        self.history.append(
            {"role": "user", "content": newMessage}
        )
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = self.history,
            temperature = self.temp
        )
        textOut = response.choices[0].message.content
        self.history.append(
            {"role": "assistant", "content": textOut}
        )
        self.latestTextOut = textOut



def main():
    currentConvo = convo()
    currentConvo.response("Hello, how are you?")
    print(currentConvo.latestTextOut)

if __name__ == "__main__":
    main()