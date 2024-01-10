import openai
import numpy as np
from utils.env import *
import requests
import base64

class Baseline():
    def __init__(self, key_file_path = './key/key.txt', VERBOSE=True):
        self.system_prompt = """
        I am a robot that can pick up objects and place them.
        Please remember that you cannot move the bowls, you can only move blocks.
        You can put blocks in a bowl, or move blocks on top of other blocks, or move blocks into a certain position.
        I have to reason out the user's preferences from the feedbacks.
        Each time, user will give a feedback about the goal. The user will stop me when the user thinks the robotic plan should be modified.
        I can spawn maximum 5 objects at a time,
        The preferences are pretty simple.
        Skills available: pick block place in bowl, pick block place on block, pick block place in next to other block
        Available objects are: {}
        """.format(ALL_BLOCKS+ALL_BOWLS)

        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.vision_messages = [{'role': 'system', 'content': self.system_prompt}]
        self.answer = None
        self.log = []
        self.generate_obj_prompt()
        api_key = self.set_openai_api_key_from_txt(key_file_path,VERBOSE=VERBOSE)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
    def set_openai_api_key_from_txt(self, key_path='./key/rilab_key.txt',VERBOSE=True):
        """
            Set OpenAI API Key from a txt file
        """
        with open(key_path, 'r') as f: 
            OPENAI_API_KEY = f.read()
        if VERBOSE:
            print("OPENAI KEY SET")
        return OPENAI_API_KEY
    
    def generate_goal_prompt(self, objects):
        self.goal_prompt = """
        The user told me what they like, but we haven't reached the goal yet. Please generate different goals that match the user's preference.
        I can see {}
        Next, specify one goal state of the environment, written in one sentence.
        Based on the user's feedback, please reason out the user's preference
        Please match the preference.
        Please try to understand the users preference from the feedback. And try to make different context of goal from the past goals, do not generate in same context.
        REMEBER: The goal should not contain any objects that are not in the environment, and different from the previous goal.
        In the goal, you have to move all the blocks in the environment, not any bowls.
        Skills available: place block in next to block. pick block place in bowl, pick block place on block
        format: 
            preference =
            goal1 = 
        """.format(str(objects))

    def generate_obj_prompt(self):
        self.obj_prompt = """
        Now, you have to build an environment that helps to find out the user's preference.
         you can have same color of bowl as block. It can have some meaning.
        Choose an objects you want to spawn in the environment. You can choose 3-5 objects.
        format: objects = []
        """

    def scene_generation(self, flag = False):
        objects = None
        if flag: template = 'user did not like the previous scene as it can not reflect the preference. please generate a new scene that can have different meaning.'
        else: template = ''
        self.messages.append({'role':'user','content':self.obj_prompt+template})
        while objects == None:
            f_src = self.ask()
            lines = f_src.split('\n')
            print(f_src)
            for l in lines:
                l = l.lower()
                if 'objects' in l:
                    objects = l

        objects = objects.split("=")[-1]
        objects = objects.split(",")
        new_objects = []
        for o in objects:
            o = o.split("'")[1]
            new_objects.append(o)
        return new_objects
    
    def append_feedback(self, feedback):
        self.messages.append({'role':'user','content':feedback})

    # def scene_image_append(self, goal, image_path= './temp/scene.png'):
        
        # question = "This is the current scene built from the user's instruction.\n{}".format(goal)
        # content = [{"type": "text", "text": question}]
        # base64_image = self.encode_image(image_path)
        # image_message = {
        #         "type": "image_url",
        #         "image_url": {
        #             "url": f"data:image/jpeg;base64,{base64_image}"
        #         }
        #     }
        # content.append(image_message)
        # user_message = {
        #     "role": "user",
        #     "content": content
        # }
        # self.vision_messages.append(user_message)

    def user_feedback_append(self, image_path = './temp/stop.png', flag = True,\
                             goal=None, prev = True):
        # content = []
        question = "This is the initial scene built from the user's instruction.\n{}".format(goal)
        content = [{"type": "text", "text": question}]
        base64_image = self.encode_image(image_path)
        image_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        content.append(image_message)
        if not flag:
            question = "The user liked the current execution, try to use this goal to infer the preference. "
            if goal != None:
                question += f"\nThis is the goal that the user liked: {goal}"
            question+= "\nThis is the final scene built from the user's previous instruction."
            content.append({"type": "text", "text": question})
        if flag:
            question = "The user stopped me here during excecuting the goal."
            question += f"\nThis is the current goal: {goal}"
            if prev:
                question += "\n This is the scene where the user preffered."
            content = [{"type": "text", "text": question}]
            if prev:
                base64_image = self.encode_image("./temp/previous.png")
                image_message = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                content.append(image_message)
            question = "But the user stopped me here. The previous preference is not correct. \
                Please generate totally differet preference. The user did not perfer the current scene."
            content.append({"type": "text", "text": question})

        base64_image = self.encode_image(image_path)
        image_message = {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }}
        content.append(image_message)
        # 
        if flag:
            task = "Let's try to analyze the images and infer the user's preference. The preference is related to the skills.\
                Concentrate only on the changes between the images, that is where the discomfort came from.\
                next to should be carefully determined with comparing with the previous image.\
                compare with the goal to extract the exact changes, that is the part the user dislike.\
                REMEMBER: Please remeber that the preference more related to the semantic meaning, not the exact precision of placing.\
                    If the user liked the current execution, try to tell me what is the undersired  part of the execution. If the user did not like the current execution, try to tell me what is the desired part of the execution.\
                    Try to **explore** the different skills compared to previous images.\
                Skills: put block next to a block, put block on a bowl, put block on a block\
                Please summarize your answer in less than three sentences based on the images and histories"
        else:
            task = "The user liked the current execution, try to use this goal to infer the preference."
        content.append({"type": "text", "text": task})
        user_message = {
            "role": "user",
            "content": content
        }
        self.vision_messages.append(user_message)
        reason = self.ask_vision()
        # self.vision_messages = [{"role": "system", "content": self.system_prompt}]
        return reason
    
    def append_fail(self):
        question = "I failed to generate the goal. Please generate the simplier goal that is similar to the previous goal."
        self.messages.append({"role": "user", "content": question})

    def ask(self):
        f_src = openai.ChatCompletion.create(
                        messages=self.messages,
                        model = 'gpt-4-1106-preview'
                    )['choices'][0]['message']['content'].strip()
        self.messages.append({"role": "assistant", "content": f_src})
        return f_src

    def ask_vision(self, APPEND=True):
        payload = self.create_payload()
        while True:
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
                print(response.json())
                res = response.json()['choices'][0]['message']['content']
                break
            except: pass
        if APPEND:
            self.vision_messages.append({"role": "assistant", "content": res})
        return res
    
    def create_payload(self):
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": self.vision_messages,
            "max_tokens": 4096,
            "temperature": 0.0,
            "n": 1
        }
        # if len(self.stop) > 0:
        #     payload["stop"] = self.stop
        return payload
    
    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def goal_generation(self):
        self.messages.append({'role':'user','content':self.goal_prompt})
        goal, preference = None, None
        while goal == None or preference == None:
            f_src = self.ask()
            lines = f_src.split('\n')
            print(f_src)
            for l in lines:
                l = l.lower()
                if 'goal' in l and "=" in l:
                    goal = l
                if 'preference' in l and "=" in l:
                    preference = l
        return goal, preference