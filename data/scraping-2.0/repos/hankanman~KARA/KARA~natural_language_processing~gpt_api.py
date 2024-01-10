```python
import openai

class GPTAPI:
    def __init__(self):
        self.api_key = "YOUR_OPENAI_API_KEY"
        openai.api_key = self.api_key

    def process(self, user_command):
        try:
            response = openai.Completion.create(
              engine="text-davinci-002",
              prompt=user_command,
              temperature=0.5,
              max_tokens=100
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error in processing user command: {e}")
            return None
```