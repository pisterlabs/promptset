import LinuxEnv as gym 
import numpy as np
import time, os
from openai import OpenAI 

from config import *
import base64
import requests

from PIL import Image


class Agent:
    def __init__(self):
        with open('base_prompt.txt', 'r') as f:
            self.base_prompt = f.read()

        with open('action_prompt.txt', 'r') as f:
            self.action_prompt = f.read()



        # initialize GPT-4 client 
        self.client = OpenAI(
            api_key=os.environ["PATH"],
        )



    def get_action(self, observation):
        print('getting agent action')
        # encode observation into base64
        # quickfix store observation as .jpeg
        print(np.shape(observation)) # (1200, 1920, 3)
        # convert to PIL image
        img = Image.fromarray(observation.astype('uint8'), 'RGB')
        # store as jpeg 
        img.save('observation.jpeg')
        #img = Image.fromarray(observation)
        #base64_image = self._encode_image()


        # call GPT-4
        print('calling GPT-4')
        response = self._call_gpt(None)
        input(response)
        print(type(response))

        # get new tokens 

        print(response.choices[0].message.content.split('Actions: ')[1].replace("\n", ""))
        output = response.choices[0].message.content.split('Actions:')[1].replace("\n", "")
        print(output)

        return output

    def _call_gpt(self, base64_image):
        time.sleep(1)
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.base_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,"
                            + base64.b64encode(open("observation.jpeg", "rb").read()).decode(
                                "utf-8"
                            ),
                            },
                        },
                        {"type": "text", "text": self.action_prompt,},
                    ],
                }
            ],
            max_tokens=300,
        )
        return response#.json()


    def _encode_image(self, img):
        img = base64.b64encode(img).decode('utf-8')





env = gym.VirtualMachineEnv(
    vnc_host="127.0.0.1",
    vnc_port=5999,
    vnc_password="password",
)

agent = Agent()

obs = env.reset()

for _ in range(50):
    action_list = agent.get_action(obs)
    # convert action_list string to list of tuples 
    action_list = eval(action_list)
    input(action_list)
    for action in action_list:
        print(f'Performing action: {action}')
        obs, _, _, _ = env.step(action)

    env.render()
    print('sleeping for 1 second')
    time.sleep(1)