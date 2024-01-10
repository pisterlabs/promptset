from openai import ChatCompletion
import openai

# Directly setting the OpenAI API key
OPENAI_API_KEY = "openai-api-key-here"
openai.api_key = OPENAI_API_KEY

class AgentHead():
    def __init__(self, n_breakups):
        self.response_schema = []
        for i in range(0, n_breakups):
            self.response_schema.append(f"Sub-task number {i} of the given task")

    def generate_response(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant and will infer what needs to be done without asking any more questions."},
                          {"role": "user", "content": prompt}]
            )
            return response.choices[0].message['content']
        except Exception as e:
            print(f"Error in generating response: {e}")
            return None

# Example usage
if __name__ == "__main__":
    agent = AgentHead(n_breakups=5)
    user_prompt = "I need to plan my day"
    response = agent.generate_response(user_prompt)
    print(response)
