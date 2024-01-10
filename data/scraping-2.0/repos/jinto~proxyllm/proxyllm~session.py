import openai


class Session:
    def __init__(
        self,
        system_prompt=None,
        engine="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
    ):
        self.model = "OpenAI"
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not system_prompt:
            self.system_prompt = "You are a helpful assistant"
        else:
            self.system_prompt = system_prompt
        self.messages = []

    def send(self, prompt=None):
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

        if prompt:
            self.messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.messages.append(response["choices"][0]["message"].to_dict())
        return response["choices"][0]["message"]
