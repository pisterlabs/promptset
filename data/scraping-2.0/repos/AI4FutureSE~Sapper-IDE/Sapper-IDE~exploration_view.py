import openai
import json

class exploration():
    def __init__(self, OpenaiKey):
        openai.api_key = OpenaiKey
        self.prompt = [{"role": "system", "content": "Are you ready?"}, {"role": "assistant", "content": "Yes, I am always ready to assist you to the best of my abilities. Just let me know how I can help you."},]
        self.context = """
        I would like you to act as a project manager. I have given you the conversation between user and assistant.
According to the conversation above, please summarize the key information.
You can only refer to the given conversation but not add extra information.
Do not pay too much attention to what the user is trying the system, but have high level abstraction for system design.
You should summarize it from three aspects:
1. Illustrate the key requirements of the user?
2. Desctibe the user's preference? For example, what the user like and what the user dislike. What should you have to do to satisfy the user's requirement and what you have not to do.
3. List the points that you have to pay attention to when implementing the system exlicitly.
You have to output the three aspects in the form of 1. Key Requirements:, 2. User Preference:, 3. Implementing Consideration:.You are expected to bullet list the key points for each aspects.
        """
        # self.conversation_file = conversation_file_name

    def chatbot(self):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=self.prompt
        )
        return response["choices"][0]["message"]

    def pre_design_view(self):
        self.prompt = self.prompt + [
            {"role": "system", "content": self.context},
        ]
        # Summarize conversation and extract user requirements
        return self.chatbot()["content"]

