from typing import Dict, List, Optional, Union
import autogen
from autogen import Agent
import queue
import openai
import os
import re
from summary_get import Character
from summary_get import vid2scene, Scene

openai.log='debug'

config_list = [
    {
       "model":"gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]
llm_config = {
    "model":"gpt-4",
    "temperature": 0,
    "config_list": config_list,
        
}

class UserProxyWebAgent(autogen.UserProxyAgent):
    def __init__(self, *args, **kwargs):
        super(UserProxyWebAgent, self).__init__(*args, **kwargs)

    def set_queues(self, client_sent_queue, client_receive_queue):
        self.client_sent_queue = client_sent_queue
        self.client_receive_queue = client_receive_queue

    # this is the method we override to interact with the chat
    def get_human_input(self, prompt: str) -> str:
        chat_messages = self.chat_messages
        last_message = self.last_message()

        if last_message["content"]:
            self.client_receive_queue.put(last_message["content"])
            reply = self.client_sent_queue.get(block=True)
            if reply == "exit":
                self.client_receive_queue.put("exit")
            return reply
        else:
            return 
        

space_x="652b5c1b43e8c47e4eb4829b"

# scene = vid2scene(space_x)
scene = Scene(
    name='Rocket Launch Chronicles', 
    imagery="A vast expanse of sky and earth, punctuated by the towering forms of SpaceX's Falcon 9 and Starship rockets. On the ground, a sea of excited spectators eagerly awaiting the moment of launch.",
      plot="In a world where space exploration is the pinnacle of human achievement, SpaceX's Starship and Falcon 9 rockets play a crucial role. Each launch represents a thrilling and high-stakes mission, with the fate of humanity's future in space hanging in the balance. The story unfolds through the eyes of a young narrator, who guides the audience through the intricacies of each launch, the technology of the rockets, and the significance of their missions. As the series progresses, the narrator becomes more involved in the launches, moving from observer to participant in these historic moments.",
      characters=[Character(name='The Narrator', description='A charismatic young individual with a deep understanding of space exploration and technology.', imagery='A young character, always seen with a rocket launch blueprint or a book about space. Wears casual attire with a SpaceX logo.', personality='Inquisitive, knowledgeable, and passionate about space exploration. Has a knack for breaking down complex information in an engaging and accessible way.'), Character(name='Starship', description='A monumental rocket that represents hope and ambition for the future of space exploration.', imagery='A towering and sleek rocket, gleaming under the sun, ready for launch.', personality='Ambitious, reliable, and awe-inspiring. Represents the spirit of human endeavour in space exploration.'), Character(name='Falcon 9', description='A smaller but equally important rocket that carries out crucial missions in space.', imagery='A high-tech rocket with powerful engines, always ready for the next launch.', personality="Determined, reliable, and resourceful. Plays a key role in humanity's exploration of space.")]
)
autogen_characters =[]

def filter_string(input_str: str) -> str:
    return "".join(re.findall("[a-zA-Z0-9_-]", input_str))

scene_characters : List[Character] = scene.characters

for character in scene.characters:
    a_c = autogen.AssistantAgent(
        name=filter_string(character.name),
        llm_config=llm_config,
        system_message=f'''You are an anime character in a scene. You are role playing as {character.name}. Description of character: {character.description}. Your personality is: {character.personality}
The name of the scene: {scene.name}

Please play your role, interact with the other characters, and craft a fun and engaging story.
Your messages will be presented like they are captions or pieces of a comic book, so keep them succinct. You are only playing the role of {character.name}, so please do not write messages for other characters.'''
    )
    autogen_characters.append(a_c)

print(scene)

#############################################################################################
# this is where you put your Autogen logic, here I have a simple 2 agents with a function call
class AutogenChat():
    def __init__(self, chat_id=None, websocket=None):
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = queue.Queue()
        self.client_receive_queue = queue.Queue()


        self.user_proxy = UserProxyWebAgent(  ###### use UserProxyWebAgent
            name="IanTheAstronaut",
            human_input_mode="ALWAYS", ######## YOU NEED TO KEEP ALWAYS
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            llm_config=llm_config,

        )
        self.assistants = autogen_characters

        # APPENDING AFTER vvv 

        autogen_characters.append(self.user_proxy)
        groupchat = autogen.GroupChat(
        agents=autogen_characters,
        messages=[],
        max_round=50,
        )
        self.manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


        # add the queues to communicate 
        self.user_proxy.set_queues(self.client_sent_queue, self.client_receive_queue)


    def set_thread(self, thread):
        self.thread = thread

    def start(self):
        data = self.client_sent_queue.get(block=True)
        self.user_proxy.initiate_chat(
            self.manager,
            message=data
        )

