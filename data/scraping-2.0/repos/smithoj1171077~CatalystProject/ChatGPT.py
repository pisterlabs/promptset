import os
import openai
openai.api_key = os.environ.get("OPENAI")

"""This class is used by all the functions of the system which use ChatGPT"""
class ChatGPT:
    def __init__(self, system_msg: str, outputFormat: str):
        self.system_msg = system_msg
        self.outputFormat = outputFormat
        self.response = ""
    def getChatGPTResponse(self, user_msg: str):
        print("used")
        self.response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=[{"role": "system", "content": self.system_msg},
                                                          {"role": "user", "content": user_msg}])
        return self.response["choices"][0]["message"]["content"]
