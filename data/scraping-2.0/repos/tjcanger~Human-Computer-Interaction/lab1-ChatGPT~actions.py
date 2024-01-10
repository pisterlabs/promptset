import openai
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from typing import Any, Text, Dict, List, Union, Optional

# openai.api_key = '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'
openai.api_key = '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'

class GPTAction(Action):
    def name(self) -> Text:
        return "action_generate"
    
    def run(self, dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # # openai.Completion.create:
        # while True:
        #     # prompt = 'Please Describe a beautiful scene including the moon, a pavilion, a lake, and a maiden.'
        #     # prompt = 'Which came first, the egg or the chicken?'
        #     prompt = input("Please enter a prompt: ")
        #     if prompt == "quit":
        #         break
        #     response = openai.Completion.create(
        #         engine="text-davinci-003",
        #         prompt=prompt,
        #         max_tokens=200,
        #         n=1,
        #         stop=None,
        #         temperature=0.9,
        #         frequency_penalty=2,
        #         presence_penalty=2,
        #     )

        #     generated_text = response.choices[0].text
        #     print(generated_text)

        # openai.ChatCompletion.create:
        history = []
        prompt="你是一家水果店的销售代表，负责辅助顾客完成购买"
        history.append({"role":"system", "content": prompt})
        hello_message = input("you: Hello!")
        history.append({"role": "user", "content": hello_message})
        first_response = 'Hello! Can I help you?'
        print("assistant:" + first_response)
        history.append({"role":"assistant", "content": first_response})

        while True:
            new_message = input("you: ")
            if new_message == 'Bye':
                break
            history.append({"role": "user", "content": new_message})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=history,
                temperature=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                max_tokens=200,
                n=1,
                stop=None
            )
            generated_text = response["choices"][0]["message"]["content"]
            history.append({"role":"assistant", "content":generated_text})
            print('assistant: '+generated_text)

        return []
    
    