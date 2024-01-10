import os
import openai


class GPT:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=2048):
        self.model_ = model
        self.max_tokens_ = max_tokens
        openai.organization =""
        openai.api_key = ""
        self.messages_ = []

    # DEAL WITH CHANGES TO MODEL
    def switch_model(self, model):
        """
        Switches the model to a new model
        """
        self.model_ = model

    @staticmethod
    def get_models():
        """
        :return: A list of all available models to use.
        """
        return openai.Model.list()

    # DEAL WITH CHANGES TO MESSAGE, SYSTEM
    def add_system_message(self, content):
        self.messages_.append({"role": "system", "content": content})

    def replace_system_message(self, content):
        self.messages_[0] = {"role": "system", "content": content}

    # DEAL WITH CHANGES TO MESSAGES, USER AND ASSISTANT

    def remove_first_k_messages(self, k):
        """
        Removes the first k messages from the messages list not including the system message
        """
        self.messages_ = self.messages_[0] + self.messages_[k:]

    def clear_messages(self):
        """
        Clears the messages list
        """
        self.messages_ = [self.messages_[0]]

    def chat(self, content):
        """

        :param content:
        :return:
        """
        self.messages_.append({"role": "user", "content": content})
        response = openai.ChatCompletion.create(model=self.model_, messages=self.messages_, temperature=0,
                                                max_tokens=self.max_tokens_)
        assistant_msg = response['choices'][0]['message']['content']
        self.messages_.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg


if __name__ == '__main__':
    gpt = GPT()
    print(gpt.chat("Hello"))
