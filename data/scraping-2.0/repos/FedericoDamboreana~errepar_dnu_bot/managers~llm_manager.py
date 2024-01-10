import openai

class LLM:
    def __init__(self, model, primer, api_key) -> None:
        openai.api_key = api_key
        self.model = model
        self.primer = primer
    
    def run(self, history, prompt):
        messages = []
        messages.append({"role": "system", "content": self.primer})
        messages = messages + history
        messages.append({"role": "user", "content": prompt})

        res = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )

        return res["choices"][0]["message"]["content"]