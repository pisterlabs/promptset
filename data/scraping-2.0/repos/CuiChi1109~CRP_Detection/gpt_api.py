import os
import openai
import json

PROM_SIMPLE = "You are an assistant, helping people with identifying phishing websites. \
                Given the words extracted from the webpage, please tell the user if the webpage is credential-requiring or not. \
                Please just give a score from 1-10, 1 is not credential, 10 is credential. Remember give nothing except a number. \
                For example, if a webpage ask user about username and password, you should score it 10"

PROM_COMPLEX = "You are a web programmer and security expert working on detecting phishing website. Now you have a task about figuring out whether a webpage is a credential-requiring page(CRP) or not. To complete this task, follow these sub-tasks: \
                Explicit: Analyze whether the webpage want users to give out sensitive information such as username, email, password, and credit card number.\
                Implicit: Analyze whether there is any button or link that linked to an explicit CRP.\
                Submit your findings as JSON-formatted output with the following keys: \
                CRP_score: int (indicates CRP on scale of 1-10) \
                CRP_type: int (0 for non-CRP,  1 for implicit, 2 for explicit) \
                Phishing_score: int (indicates phishing risk on scale of 1-10) \
                Remember give nothing except a JSON-formatted output. \
                Noted, if you cannot distinguish based on the given text, please treat it as non-CRP. \
                Example of credential-requiring: \
                Having inputs fields about username, password, credit card number or buttons to login \
                Offer unexpected rewards \
                Informing the user of a missing package or additional payment required \
                Displaying fake security warnings \
                This is the text extracted from screenshots of webpages. \ "

# PROM_EXAMPLE = "You are an assistant, helping people with identifying phishing websites. \

prompts_dict = {
    "simple": PROM_SIMPLE,
    "complex": PROM_COMPLEX
}

openai.api_key = os.environ["OPENAI_API_KEY"]

class GPTConversationalAgent:
    def __init__(self, model, prompts=PROM_SIMPLE):
        self.messages = [
            {
                "role": "system",
                "content": prompts
            }
        ]
        self.model = model

    def call_gpt(self, text, image=None):
        if image != None:
            print(f'Call gpt on {image}')
        self.messages.append({"role": "user", "content": text})
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=0.0
        )
        response = completion.choices[0].message.content
        self.messages.pop()     # delete the user ask
        return response