import openai
import re

class Bookgpt:
    def __init__(self, message_history):
        openai.api_key = "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"
        self.message_history = message_history
    
    def predict(self,input):
        # tokenize the new input sentence
        self.message_history.append({"role": "user", "content": f"{input}"})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", #10x cheaper than davinci, and better. $0.002 per 1k tokens
            messages= self.message_history
        )
        #Just the reply:
        reply_content = completion.choices[0].message.content#.replace('```python', '<pre>').replace('```', '</pre>')
        print(reply_content)
        print(type(reply_content))
        self.message_history.append({"role": "assistant", "content": f"{reply_content}"}) 
        
        # get a list of reply_content 
        # delete number and punctuation
        reply_content = re.sub('[0-9.]+', '', reply_content)
        response = list(reply_content.split("\n"))
        response = list(map(lambda x: x.strip('"\' \n\t'), response))
       
        if len(response) > 5:
            response = response[2:]
        elif len(response) == 1:
            response = []

        print(response)
        print(type(response))
        
        self.message_history.pop(-1)
        self.message_history.pop(-1)
        
        return response

