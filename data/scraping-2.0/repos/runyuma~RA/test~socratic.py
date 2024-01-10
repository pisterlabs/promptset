import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward
from tasks.letter import PutLetterOontheBowl,PutLetterontheBowlUnseen,PutLetterontheBowl,PutLetterontheBowlUnseenLetters
import numpy as np
import matplotlib.pyplot as plt
from agents.LLMRL import LLMSAC,GuideSAC
import cv2

model_seed = 3
model_name = "tmp/llmsac_imgatten_withnoise_PutLetterontheBowlseed"+str(model_seed)+"_model/rl_model_60000_steps.zip"
model =LLMSAC.load(model_name)
file = open("key", "r")
openai_api_key = file.read()
import openai
openai.api_key = openai_api_key
gpt3_prompt = """
objects = ["cyan block", "yellow block", "brown block", "green bowl"]
# move all the blocks to the top left corner.
robot.pick_and_place("brown block", "top left corner")
robot.pick_and_place("cyan block", "top left corner")
robot.pick_and_place("yellow block", "top left corner")
# put the yellow one the green thing.
robot.pick_and_place("yellow block", "green bowl")
# undo that.
robot.pick_and_place("yellow block", "top left corner")
objects = ["pink block", "gray block", "orange block"]
# move the pinkish colored block on the bottom side.
robot.pick_and_place("pink block", "bottom side")
objects = ["orange block", "purple bowl", "cyan block", "brown bowl", "pink block"]
# stack the blocks.
robot.pick_and_place("pink block", "orange block")
robot.pick_and_place("cyan block", "pink block")
# unstack that.
robot.pick_and_place("cyan block", "bottom left")
robot.pick_and_place("pink block", "left side")
objects = ["red block", "brown block", "purple bowl", "gray bowl", "brown bowl", "pink block", "purple block"]
# group the brown objects together.
robot.pick_and_place("brown block", "brown bowl")
objects = ["orange bowl", "red block", "orange block", "red bowl", "purple bowl", "purple block"]
# sort all the blocks into their matching color bowls.
robot.pick_and_place("orange block", "orange bowl")
robot.pick_and_place("red block", "red bowl")
robot.pick_and_place("purple block", "purple bowl")
"""

gpt_version = "text-davinci-002"
def LM(prompt, max_tokens=128, temperature=0, stop=None):
  response = openai.Completion.create(engine=gpt_version, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
  return response["choices"][0]["text"].strip()
def act(prompt_response):
    print(prompt_response)
    actions = prompt_response.split("\n")
    action = []
    for act in actions:
        if "robot.pick_and_place" in act:
            # get before the first ( and after the last )
            act = act.split("(")[1]
            act = act.split(")")[0]
            pick,place = act.split(",")
            act = [pick[1:-1],place[2:-1]]
            action.append(act)
    return action
def single_step(env,actions,obs,step_limit = 4):
    dones = []
    for act in actions:
        done = False
        step = 0
        pick_idx = env.task.config["pick"].index(act[0])
        place_idx = env.task.config["place"].index(act[1])
        obs["lang_goal"] = np.array([pick_idx,place_idx])
        env.task.goals = [pick_idx,place_idx]
        while not done:
            _act,_ = model.predict(obs, deterministic=True)
            obs, rewards, done,_, info = env.step(_act)
            print(_act,rewards,done)            
            step += 1
            if step >= step_limit:
                done = True
            if done:
                success = (rewards >= 0.9)
        dones.append(success)
    if np.mean(dones) ==1:
        print("success")
        return True
    else:
        print("fail")
        return False
            

    

user_input = 'put all the blocks on the bowl'
env = ResPickOrPlaceEnvWithoutLangReward(
                                        task= PutLetterontheBowl,
                                         image_obs=True,
                                         residual=True,
                                         observation_noise=5,
                                         render=True,
                                         multi_discrete=False,
                                         scale_action=True,
                                         ee="suction",
                                         scale_obs=True,
                                         neglect_steps=False,
                                      one_hot_action = True)
trail = 1 #success rate 50%
successes = []
for i in range(trail):
    np.random.seed(4)
    obs,_ = env.reset()
    scene = env.task.config
    scene_description = "objects = ["
    for obj in scene["pick"]:
        scene_description += "\"" + obj + "\", "
    for obj in scene["place"]:
        scene_description += "\"" + obj + "\", "
    scene_description = scene_description[:-2]
    scene_description += "]"
    context = gpt3_prompt
    context += scene_description + '\n'
    context += '# ' + user_input + '\n'
    response = LM(context, stop=['#', 'objects ='])
    context += response + '\n'
    actions = act(response)
    success = single_step(env,actions,obs)
    successes.append(success)
print(np.sum(successes))


   
