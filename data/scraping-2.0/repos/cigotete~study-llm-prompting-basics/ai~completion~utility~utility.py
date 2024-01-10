import openai
import os


class AIUtility:
    openai.api_key = os.getenv("OPENAI_KEY")

    def __init__(self, prompt, temperature=0, model="gpt-3.5-turbo-1106"):
        self.prompt = prompt
        self.temperature = temperature
        self.model = model

    def __str__(self):
        return f"{self.prompt}"

    def get_completion(self):
        messages = [{"role": "user", "content": self.prompt}]
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            seed=864,
        )
        return response.choices[0].message.content.strip()

    def print_completion(self):
        # Use a breakpoint in the code line below to debug your script.
        response = self.get_completion()  # Press ⌘F8 to toggle the breakpoint.
        print(response)
        print("-----------------------------------------------------------------")