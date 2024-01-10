import openai
import os
import json
from json import JSONDecodeError
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


class SituationAction:
    def __init__(self, situation, valid_actions, call_to_action, action_format):
        self.situation = situation
        self.valid_actions = valid_actions
        self.call_to_action = call_to_action
        self.action_format = action_format
        self.initial_action = ""
        self.validated_action = ""
        self.initial_format_validated_action = ""

    def get_action_prompt(self):
        return """
This is the situation: {situation}
These are the set of valid actions to take: {valid_actions}
{call_to_action}
        """.format(situation=self.situation, valid_actions=self.valid_actions, call_to_action=self.call_to_action)

    def valid_action_check(self):
        return """
Given the situation: {situation}
And the action you chose: {initial_action}
Is the action you in this set of valid actions: {valid_actions}?
If not, choose the best valid action to take. If so, please return the original action
        """.format(situation=self.situation, initial_action=self.initial_action, valid_actions=self.valid_actions)

    def get_action_format(self):
        return """
This is the correct format for an action: {action_format}
This is the chosen action: {validated_action}
Convert the chosen action to the correct format.
        """.format(action_format=self.action_format, validated_action=self.validated_action)

    def get_valid_format(self):
        return """
This is the correct format for an action: {action_format}
This is a formatted action: {initial_format_validated_action}
Return the action in the correct format.
        """.format(action_format=self.action_format,
                   initial_format_validated_action=self.initial_format_validated_action)

    def run(self):
        prompt = self.get_action_prompt()
        print(prompt)
        # 1.INTIAL ACTION
        messages = [{"role": "user",
                     "content": prompt}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.7)
        reply = completion.choices[0].message.content
        print(reply)
        self.initial_action = reply
        # 2.CHECK VALID ACTION
        prompt = self.valid_action_check()
        print(prompt)
        messages = [{"role": "user",
                     "content": prompt}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.7)
        reply = completion.choices[0].message.content
        print(reply)
        self.validated_action = reply
        # 3.FORMAT ACTION TO JSON
        prompt = self.get_action_format()
        print(prompt)
        messages = [{"role": "user",
                     "content": prompt}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.7)
        reply = completion.choices[0].message.content
        try:
            res_json = json.loads(reply)
            print(res_json)
        except JSONDecodeError as e:
            # 4.VERIFY FORMAT
            self.initial_format_validated_action = reply
            prompt = self.get_valid_format()
            print(prompt)
            messages = [{"role": "user",
                         "content": prompt}]
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.7)
            reply = completion.choices[0].message.content
            print(reply)


situation = """
Let's play poker. Your name is Tommy and you are a player in a game of No-Limit Texas Hold'em Poker. 
You have the cards Ac, Ah. The board is []. You have $100 in your stack. 
The pot is $20. You need to put $3 into the pot to play. 
The current bet is $3, and you are in seat 9 out of 9. 
Your position is in the Cutoff.
"""
valid_actions = """
You can call for $5, raise between $6 and $100, or fold for $0
"""

call_to_action = """
What is the action you would like to take out of the following: ('call', 'raise', 'fold')? 
"""

action_format = """
Specify the amount associated with that action in the format:
{
    action: {
        reason: string,
        type: string
    }
    amount: number
}
ONLY return values in this format (no other text is necessary)
"""
s = SituationAction(situation, valid_actions, call_to_action, action_format)
s.run()
