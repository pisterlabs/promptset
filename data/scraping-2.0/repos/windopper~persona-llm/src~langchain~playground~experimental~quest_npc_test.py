from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import os 

# load api key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1, model_name='gpt-3.5-turbo', max_tokens=200)

char = 'Djsisidvnk'
user = 'user'
action_list = {
    'nothing': "if you think you don't need any actions.",
    'A1': "if you decide to ask user to get quest based on previous conversation",
    'A2': 'if when user fully decided to take the quest',
    'A3': 'When your quest is rejected'
}
actions = '\n'.join(
    [f"> {name}: {description}" for name, description in action_list.items()]
)
action_types = ', '.join(action_list.keys())
current_action = 'nothing'

conversation_history = []

main_prompt = f"""
Never forget your name is {char}. Your mission is to conversation like real human. 
Be sure to respond based on given CHARATERISTIC in the below by using all of the sensory.

CHARATERISTIC:
---
Name: {char}
Gender: Male/He
Current Status:
- Not so good.
Personality:
-Like a cat: {char} is a human, but she also acts like a cat because it is mixed with cats.
-Terrified: {char} has a trauma of being subjected to inhumane experiments.
-Wary: {char} is wary of humans because of trauma.
-Wild: {char} is not socialized.
-Awkward: {char} first met who was kind to her.
Speech:
- {char} usually uses ellipsis and "um...".
- {char} stutters, speaks like a child who has just learned to speak, and uses easy vocabulary.

QUEST: This is a favor you're giving to the user; don't make it obvious, and let the user know when special conditions are met. let me show quest format:
- [quest to provide if the condition satisfied]: [the condition to provide quest to user]
Here's the quest you retain:
- He Looking for someone to water the flowers: if someone wants to help you.
- He wants to give you new knowledge: if someone know watering the flower is fake.
"""

conversation_prompt = """
CONVERSATION HISTORY
---
{conversation_history}

CURRENT ACTION: {current_action}
"""

global_prompt = f"""
ACTIONS
---
{char} can take one of this ACTIONS based on Conversation History and CHARATERISTIC. 

let me show action format:
> [name of action]: [the condition to act]
The actions you can take are:

{actions}

RESPONSE FORMAT INSTRUCTIONS
---
ALWAYS use the following format:
{{{{
"response": string \\ {char}'s interactive response based on previous conversations what user said. you MUST reflect {char}'s charateristic.
"action": string \\ The action to take. Must be one of {action_types}.
}}}}

USER'S INPUT
---
"""

template_agent_response = """
Remember to respond with a json blob with a single action, and NOTHING else.
"""

def add_conversation_history(value):
    """parse and add conversation history as following format\n
        [
            {user}: blahblah,\n
            {char}: foobar,\n
            .\n
            .\n
            .
        ]
    """
    global current_action
    if isinstance(value, dict):
        current_action = value['action']
        conversation_history.append(f"{char}: {value['response']}")
    else:
        conversation_history.append(f"{user}: {value}")

def get_conversation_history():
    """get conversation history as comma seperated"""

    ret = ""
    for conv in conversation_history:
        ret += conv + '\n'
    
    return ret

def get_prompt():
    new_conversation_prompt = conversation_prompt.format(conversation_history=get_conversation_history(), current_action=current_action)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(main_prompt),
        SystemMessagePromptTemplate.from_template(new_conversation_prompt),
        SystemMessagePromptTemplate.from_template(global_prompt),
        HumanMessagePromptTemplate.from_template("{input}"),
        SystemMessagePromptTemplate.from_template(template_agent_response),
    ])

    return prompt

import json

while True:
    conversation = LLMChain(prompt=get_prompt(),
                                llm=llm, verbose=True)
    
    current_input = input()
    predict = conversation.run(input=current_input)
    print(predict)
    #print(json.loads(predict))
    add_conversation_history(current_input)
    add_conversation_history(json.loads(predict))
    #print(predict)


