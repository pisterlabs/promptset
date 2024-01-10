from phue import Bridge
import openai
from dotenv import load_dotenv
from os import getenv
import json


def cost_calc(num_tokens: int) -> float:
    """
    For parm number of tokens used, return cost incurred in USD.
    """
    # From, https://openai.com/pricing, gpt-3.5-turbo is $0.002 per 1000 tokens.
    return num_tokens * 0.002 / 1000


class Persona:

    def __init__(self, gpt_model: str, persona_name: str, functions: list):

        self.gpt_model = gpt_model
        self.persona_name = persona_name
        self.functions = functions
        self.history = []
        self.cumulative_tokens = 0

    def give_mission(self, mission: str, response: str):
        print(self.persona_name + ' mission...')
        print(mission)
        print('------------')
        self.update_history(role='user', content=mission)

        # 'Trick' GPT into thinking it understood us earlier in the conversation.
        self.update_history(role='assistant', content=response)

    def update_history(self, role: str, content: str):
        assert role in ['assistant', 'user']
        self.history.append({'role': role, 'content': content})

    def chat(self, prompt: str):
        self.update_history(role='user', content=prompt)

        completion = openai.ChatCompletion.create(model=self.gpt_model,
                                                  messages=self.history,
                                                  functions=self.functions,
                                                  function_call="auto")

        self.cumulative_tokens += int(completion.usage.total_tokens)
        reply_content = completion.choices[0].message
        # print(reply_content)

        if 'function_call' in reply_content:
            func_name = reply_content['function_call']['name']
            args_json = reply_content.to_dict()['function_call']['arguments']
            payload = json.loads(args_json)
            payload['function_name'] = func_name
            # print(payload)
            return payload
        else:
            content = reply_content['content']
            print(self.persona_name + ': ' + content)
            self.update_history(role='assistant', content=content)
            return {}


class Lights:

    def __init__(self, bridge_ip):
        self.bridge = Bridge(bridge_ip)

        # If the app is not registered and the button is not pressed, press the button and call connect()
        # (this only needs to be run a single time)
        self.bridge.connect()

        # Get the bridge state (This returns the full dictionary that you can explore)
        bridge_state = self.bridge.get_api()

        # Make a single dictionary of all the lights and groups.
        # Key = name of individual light or group of lights, value = list of light IDs.
        self.lights = {}

        for light_id in bridge_state['lights']:
            light_name = bridge_state['lights'][light_id]['name']
            self.lights[light_name] = [light_id]

        remove_later = set()

        for group_id in bridge_state['groups']:
            group_name = bridge_state['groups'][group_id]['name']

            for candidate in self.lights:
                if group_name in candidate:
                    remove_later.add(candidate)

            self.lights[group_name] = bridge_state['groups'][group_id]['lights']

        # Remove individual lights that have names that are substrings of groups.
        for each_light in remove_later:
            del self.lights[each_light]

    def describe_lights(self) -> list:
        """
        Generate a list of the lights in the house, suitable for telling the chatbot about.
        """
        return [name for name in self.lights]

    def turn_on_or_off(self, light_name: str, on: bool):
        if light_name not in self.lights:
            print('Light not found.')
            return

        # Adjust all of the lights in the list.
        for light_id in self.lights[light_name]:
            self.bridge.set_light(int(light_id), 'on', on)

        if on:
            print('Turned on ' + light_name)
        else:
            print('Turned off ' + light_name)

    def set_brightness(self, light_name: str, brightness: int):
        if light_name not in self.lights:
            print('Light not found.')
            return

        # Adjust all of the lights in the list.
        for light_id in self.lights[light_name]:
            self.bridge.set_light(int(light_id), {'on': True, 'bri': brightness})

        print(light_name + ' set to ' + str(int(100 * brightness / 254)) + '% brightness')

    def interpret_response(self, gpt_response: dict):
        """
        Interpret the response from the chatbot.
        """
        if len(gpt_response) == 0:
            return
        # print(gpt_response)
        if gpt_response['function_name'] == 'turn_on_or_off':
            self.turn_on_or_off(light_name=gpt_response['light_name'], on=gpt_response['on'])
        elif gpt_response['function_name'] == 'set_brightness':
            self.set_brightness(light_name=gpt_response['light_name'], brightness=gpt_response['brightness'])


load_dotenv(verbose=True)           # Set operating system environment variables based on contents of .env file.
my_lights = Lights(getenv('BRIDGE_IP'))
lights_list = my_lights.describe_lights()
print('lights_list:', lights_list)

# my_lights.turn_on_or_off(light_name='Master Bedroom', on=True)
# my_lights.set_brightness(light_name='Master Bedroom', brightness=200)


hue_functions = [
{
    "name": "turn_on_or_off",
    "description": "Turn a Hue light bulb on or off.",
    "parameters": {
        "type": "object",
        "properties": {
            "light_name": {
                "type": "string",
                "description": "The name of the Hue bulb that this function will turn on or off.",
                "enum": lights_list
            },
            "on": {"type": "boolean",
                   "description": "True if the light should be turned on. False if the light should be turned off."
                   }
        },
        "required": ["light_name", "on"]
    }
},
    {
        "name": "set_brightness",
        "description": "Change the level of brightness of a Hue bulb. Don't use this function for turning lights off.",
        "parameters": {
            "type": "object",
            "properties": {
                "light_name": {
                    "type": "string",
                    "description": "The name of the Hue bulb that this function will turn on or off.",
                    "enum": lights_list
                },
                "brightness": {"type": "integer",
                               "description": "The brightness that the bulb should be set to. Expressed as an integer between 0 and 254, where 0 is dark and 254 is maximum brightness.",
                               "enum": list(range(255))
                       }
            },
            "required": ["light_name", "brightness"]
        }
    }
]

openai.api_key = getenv('OPEN_AI_KEY')
chatgpt = Persona(gpt_model=getenv('OPEN_AI_MODEL'), persona_name='ChatGPT', functions=hue_functions)

mission = '''I'd like you to control the Philips Hue light bulbs in my house.
Only use the set_brightness function for changing brightness. Make sure you use the turn_on_or_off function for actually turning the lights on and off.
Please say "OK" now if you understand.'''
response = 'OK.'
chatgpt.give_mission(mission=mission, response=response)

while True:
    inp = input("User input (or 'quit'): ")
    if inp == 'quit':
        break
    resp = chatgpt.chat(prompt=inp)
    my_lights.interpret_response(gpt_response=resp)

print('\nTotal tokens used:', chatgpt.cumulative_tokens)
print('Cost incurred (USD):', cost_calc(chatgpt.cumulative_tokens))
