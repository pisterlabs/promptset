import openai
from .agent import Agent


class StatelessAgent(Agent):
    def __init__(self, api_key):
        super().__init__(api_key)

    def query(self, context):
        response = openai.ChatCompletion.create(
            model = self.model,
            messages=context.get_messages()
        )
        print(context.get_messages())
        return response["choices"][0]["message"]['content']

        # response = self.agent.run(prompt)
        # return response

