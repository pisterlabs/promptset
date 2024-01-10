```python
import openai
from openai.api_resources import Completion

class GPTGenius:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate_code(self, prompt):
        response = Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100)
        return response.choices[0].text.strip()

    def analyze_code(self, code):
        # This is a placeholder. Actual implementation will require a more complex model.
        return "This code looks good!"

    def interactive_editor(self):
        while True:
            user_input = input("Enter your code or type 'exit' to quit: ")
            if user_input.lower() == 'exit':
                break
            print(self.analyze_code(user_input))

if __name__ == "__main__":
    gpt_genius = GPTGenius("your-api-key")
    while True:
        user_input = input("Enter a prompt or type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        print(gpt_genius.generate_code(user_input))
```
