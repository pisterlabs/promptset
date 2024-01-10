```python
from openai import GPT3LM

class LanguageModel:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model = GPT3LM(model_name)

    def generate_response(self, prompt, max_tokens=100):
        response = self.model.generate(prompt, max_tokens=max_tokens)
        return response

    def generate_interview_questions(self, context):
        prompt = f"{context}\nWhat would be your next question?"
        question = self.generate_response(prompt)
        return question

    def generate_narrative(self, interviewData):
        prompt = f"Based on the following interview data: {interviewData}\nGenerate a narrative:"
        narrative = self.generate_response(prompt, max_tokens=16385)
        return narrative
```