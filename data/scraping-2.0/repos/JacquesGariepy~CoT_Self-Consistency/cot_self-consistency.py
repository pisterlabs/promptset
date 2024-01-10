import litellm
from langchain.prompts import PromptTemplate

class ChainOfThought:
    def __init__(self, model="gpt-3.5-turbo", token_limit=150):
        self.litellm = litellm.LiteLLM()
        self.model = model
        self.token_limit = token_limit

    def prompt(self, initial_prompt, iterations=5):
        template = PromptTemplate(f"{initial_prompt}, think step-by-step")
        current_prompt = template.render()
        response = self.litellm.completion(
            model=self.model,
            prompt=current_prompt,
            max_tokens=self.token_limit,
            n=iterations,
        )
        responses = [choice['message']['content'] for choice in response['choices'] if choice['message']['role'] == 'assistant']
        return ", ".join(responses)

# Usage
thought_chain = ChainOfThought()
print(thought_chain.prompt("Explain the process of photosynthesis"))

