import openai

openai.api_key = "API Key"
model_engine = "text-davinci-003"
prompt = 'Give me 100 movie names'

class Engine:
    def __init__(self, engine, prompt, max_tokens, n, stop, temperature):
        self.model_engine = engine
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop
        self.temperature = temperature

    def Fetch(self):
        completion = openai.Completion.create(
        self.model_engine,
        self.prompt,
        self.max_tokens,
        self.n,
        self.stop,
        self.temperature)

        self.response = completion.choices[0].text

    def save(self):
        text_file = open("content.txt", "w")
        n = text_file.write(self.response)
        text_file.close()
        return self.response

Cont = Engine(engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5)
