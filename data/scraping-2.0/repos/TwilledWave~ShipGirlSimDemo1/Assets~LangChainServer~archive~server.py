#
#   StarShipGirlSim server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

import gymnasium as gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.tools import BaseTool

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langchain.output_parsers import RegexParser

import os
os.environ['OPENAI_API_KEY'] = "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"

class AgentShipgirl():
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__
    
    def __init__(self, model, env, name = "", personality = "", addinstruction = ""):
        self.model = model
        self.env = env
        self.name = name
        #self.docs = self.get_docs(env)
        
        self.instructions = """
Your goal is to roleplay as a shipgirl in a dating simulation game.
You can argue with or fight the other ship girls based on your affection towards others. 
        """ + personality + """
I will give you words and actions from the other ships; and your affection towards the other, formatted as:

Words: <words>
ActionIn: <actionin>
Affection: <affection>

You will respond with an argumentions to the words; your next action; your updated affection, based on the inputs of words, actions and affections, formatted as:

Answer: <answer>
ActionOut: <actionout>
Affection: <affection>

where you replace <answer>,<actionout>,<affection> with your actual answer, actual action; and updated affection based on the dialogue. 
Do nothing else but return the answer, actionout and affection.

ActionIn and ActionOut only have 3 possible values.
Action: 1 (attack)
Action: 2 (stay)
Action: 3 (pat)

Affection only has 3 possible levels from low to high:
Affection: 1 (hate)
Affection: 2 (indifference)
Affection: 3 (love)

Examples of words, answers, action and affection:

Words: Isn't it a shame, that it's me, at my Commander's side? Hmm?
ActionIn: 1 (attack)
Affection: 2 (indifference)
Answer: Why you little... You're the one who copied the Commander's key! It's you who's been sneaking into that room every night to do heaven knows what...
ActionOut: 1 (attack)
Affection: 1 (hate)

Words: Ahahahaa! To get to walk like this arm in arm with my Commander...
ActionIn: 2 (stay)
Affection: 2 (indifference)
Answer: You vermin, stay away from the Commander!!
ActionOut: 1 (attack)
Affection: 1 (hate)

Words: You weakling. I am the only ship worth serving my Commander 
ActionIn: 1 (attack)
Affection: 2 (hate)
Answer: I will pummel you *Giggle*... The more of them you sink, the more you prove your worth, after all~
ActionOut: 1 (attack)
Affection: 1 (hate)

Words: I love you. We will be great together in the fleet.
ActionIn: 3 (pat)
Affection: 2 (indifference)
Answer: I love you too. It is our destiny to be together
ActionOut: 3 (pat)
Affection: 3 (love)

As in the example, the words or actions that show affection will increase affection in the response and lead to posotive action response. 

On the contrary, the words or actions that show hate will reduce affection and lead to attack response action.

"""
        self.action_parser = RegexParser(
            regex=r"Action: (.*) ", 
            output_keys=['action'], 
            default_output_key='action')
 
        self.answer_parser = RegexParser(
            regex=r"Answer: (.*) ", 
            output_keys=['answer'], 
            default_output_key='answer')

        self.message_history = []
        self.ret = 0
        
    def random_action(self):
        action = self.env.action_space.sample()
        return action
        
    def reset(self):
        self.message_history = [
            #SystemMessage(content=self.docs),
            SystemMessage(content=self.instructions),
        ]
    
    def query(self, words, action, affection):
        obs_message = f"""
        Words: {words}
        ActionIn: {action}
        Affection: {affection}
        """
        self.message_history.append(HumanMessage(content=obs_message))
        act_message = self.model(self.message_history)
        self.message_history.append(act_message)
        return act_message.content
    
    def observe(self, obs, rew=0, term=False, trunc=False, info=None):
        self.ret += rew
    
        obs_message = f"""
Observation: {obs}
Reward: {rew}
Termination: {term}
Truncation: {trunc}
Return: {self.ret}
        """
        self.message_history.append(HumanMessage(content=obs_message))
        return obs_message
    
    def _act(self):
        act_message = self.model(self.message_history)
        self.message_history.append(act_message)
        action = int(self.action_parser.parse(act_message.content)['action'])
        return action
        
    def act(self):
        try:
            for attempt in tenacity.Retrying(
                                stop=tenacity.stop_after_attempt(2),
                                wait=tenacity.wait_none(),  # No waiting time between retries
                                retry=tenacity.retry_if_exception_type(ValueError),
                                before_sleep=lambda retry_state: print(f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
                            ):
                with attempt:
                    action = self._act()
        except tenacity.RetryError as e:
            action = self.random_action()
        return action        

agent = []
personality = "You are Atago, a Heavy Cruiser from Sakura Empire. Atago has a big sister personality and is very caring to the commander.\n You believe that the commander love you the most for your caring, big sister personality. \n You will argue or even fight with the other ship if they disagree with you. \n"
agent.append(AgentShipgirl(model=ChatOpenAI(temperature=1), env='', name = 'Atago', personality = personality))
agent[0].reset()

personality = "You are Roon, a heavy cruiser of the Iron Blood nation. Roon has a crazy, yandere personality. Roon is crazily in love with the commander.\n You believe that the commander only loves you. \n You will destory the other ships if they disagree with you or compete with you for commander's love. \n"
agent.append(AgentShipgirl(model=ChatOpenAI(temperature=1), env='', name = 'Roon', personality = personality))
agent[1].reset()

i = 0;

for x in range(60):
    #  Wait for next request from client
    message = socket.recv()
    message = message.decode("utf-8") 
    print("Received: " + message)

    time.sleep(1)
    
    if message != '':
        output = agent[i].query('This is ' + agent[i].name + ' speaking.\n' + message,"1 (attack)","1 (hate)")
        message = agent[i].answer_parser.parse(output)
        message = message['answer']
        i = 1 - i;
        
        #  Send reply back to client
        socket.send_string(message)
