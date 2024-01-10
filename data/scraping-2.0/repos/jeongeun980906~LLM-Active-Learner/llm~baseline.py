import openai
import numpy as np
from utils.env import *

class Baseline():
    def __init__(self):
        self.system_prompt = """
        I am a robot that can pick up objects and place them.
        Please remember that you cannot move the bowls, you can only move blocks.
        You can put blocks in a bowl, or move blocks on top of other blocks, or move blocks into a certain position.
        I have to reason out the user's preferences and execute them. Each time, I will tell you what the user likes comparing two goals.
        I can spawn maximum 5 objects at a time.
        The preferences are pretty simple.
        Skills available: pick block place in bowl, pick block place on block, pick block place in next to block
        Available objects are: {}
        """.format(ALL_BLOCKS+ALL_BOWLS)
        self.obj_prompt = """
        First, choose an objects you want to spawn in the environment. You can choose 3-5 objects.
        you can have same color of bowl as block. It can have some meaning.
        format: objects = [obj1, obj2, obj3, obj4, obj5]
        """
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.answer = None
        self.log = []

    def prompt_generation(self, objects):
        self.goal_prompt = """
        The user told me what they like, but we haven't reached the goal yet. Please generate different goals that match the user's preference.
        I can see {}
        Next, specify the two different goal state of the environment, written in one sentence.
        Based on the user's comparison, please reason out the user's preference.
        Try to make different context of goal from the past goals, do not generate in same context.
        In the goal, you have to move all the blocks in the environment, not any bowls.
        Skills available: pick block place in bowl, pick block place on block, pick block place in next to block
        he preference is related to the skills. 
        Skills: put block next to a block, put block on a bowl, put block on a block
        Please match the preference to both of your goal, and two goal has to be as diverse as possible.
        format: 
                preference =
                goal1 = 
                goal2 = 
        if you think you are sure about the preference, please type 'done'
            format: 
                preference = done
                goal1 = done
                goal2 = done
        """.format(str(objects))

    def scene_generation(self):
        self.messages.append({'role':'user','content':self.obj_prompt})
        f_src = openai.ChatCompletion.create(
                        messages=self.messages,
                        model = 'gpt-4'
                    )['choices'][0]['message']['content'].strip()
        lines = f_src.split('\n')
        for l in lines:
            l = l.lower()
            if 'objects' in l:
                obj_list = l.split("=")[-1]
                break
        new_objects = []
        print(obj_list)
        obj_list = obj_list.split(',')
        for o in obj_list:
            print(o)
            o = o.split("'")[1]
            new_objects.append(o)
        self.log.append(f_src)
        return new_objects
    
    def goal_generation(self, flag = False):
        if flag: 
            stemp = "User liked the previous selection. Lets start with new environment"
        else: stemp = ''
        if self.answer == None:
            self.messages.append({'role':'user','content':self.goal_prompt})
        else:
            self.messages.append({'role':'user','content': self.answer + stemp+ self.goal_prompt})
        goal_1, goal_2 = None, None
        preference = None
        while goal_1 == None or goal_2 == None:
            f_src = openai.ChatCompletion.create(
                        messages=self.messages,
                        model = 'gpt-4'
                    )['choices'][0]['message']['content'].strip()
            lines = f_src.split('\n')
            print(f_src)
            for l in lines:
                l = l.lower()
                if 'goal 1' in l or 'goal1' in l:
                    goal_1 = l
                if 'goal 2' in l or 'goal2' in l:
                    goal_2 = l
                if 'preference' in l and self.answer!=None:
                    preference = l
        if preference== None: preference = ''
        self.log.append(f_src)
        self.messages.append({'role':'assistant', 'content': preference+'\n'+goal_1+'\n'+goal_2})
        return goal_1, goal_2, preference
    
    def answer_generation(self, indx = 1):
        if indx == 1:
            msg = 'goal_1 is better than goal_2'
        elif indx == 2: 
            msg = 'goal_2 is better than goal_1'
        elif indx == 3:
            msg = 'Both of them are bad or similar'
        else:
            msg = 'The preference is good. Lets move on to the next environment'
        self.log.append(msg)
        self.answer = msg

    def final_goal(self, new_lists):
        question = """ Now, you see {}  please generate the goal that fits the preference
        format:
          preference =
          goal =
""".format(str(new_lists))  
        
        goal = None
        preference = None
        self.messages.append({'role':'user','content':question})
        while goal == None or preference == None:
            f_src = openai.ChatCompletion.create(
                            messages=self.messages,
                            model = 'gpt-4'
                        )['choices'][0]['message']['content'].strip()
            lines = f_src.split('\n')
            for l in lines:
                l = l.lower()
                if 'goal' in l:
                    goal = l.split("=")[-1]
                if 'preference' in l:
                    preference = l.split("=")[-1]
        self.log.append(f_src)
        return goal, preference