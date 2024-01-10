import openai
from config import model
import logging

class RawModel():
    def start_conversation(self, problem, question, code):
        self.conversation = []
        prompt = "I have been given the following instructions:\n"\
            + problem + "\n\n"\
            + "I have written the following code:\n"\
            + code + "\n\n"\
            + question
        
        return self.__chat(prompt)
    
    def send_prompt(self, prompt):
        return self.__chat(prompt)

    def __chat(self, prompt):
        self.conversation.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=model,
            messages=self.conversation
        )
        responseText = response['choices'][0]['message']['content']
        logging.info("Prompt: " + prompt)
        logging.info("Response: " + responseText)
        self.conversation.append({"role": "assistant", "content": responseText})
        return responseText