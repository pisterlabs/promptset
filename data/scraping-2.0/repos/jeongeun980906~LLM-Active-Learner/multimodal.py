from utils.env import *
from utils.lmp import *
from utils.lmp_wrapper import *
import copy
from utils.cap_prompts import *
import matplotlib.pyplot as plt
from utils.config import *
from moviepy.editor import ImageSequenceClip
# import openai
# from llm.gui import GUI
from llm.baseline2 import Baseline
import openai
from llm.gui2 import GUI
import time
from PIL import Image
from utils.key_register import set_openai_api_key_from_txt
import argparse

set_openai_api_key_from_txt(key_path='./key/key.txt',VERBOSE=True)

high_resolution = False #@param {type:"boolean"}
high_frame_rate = False #@param {type:"boolean"}

def main(args):
    # setup env and LMP
    env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
    # display env

    logger = Baseline()

    gui = GUI()
    gui.root.update()
    flag = False
    num_iter = 0
    while num_iter < 5:
        if not flag or gui.label == 3:
            obj_list = logger.scene_generation(gui.label == 3)
            logger.generate_goal_prompt(obj_list)
            inp = input('continue?')
            if inp == 'n': num_iter+= 1;gui.label = 3; continue
        _ = env.reset(obj_list)
        goal_1, preference = logger.goal_generation()
        goal_1 = goal_1.split("=")[-1]
        preference = preference.split("=")[-1]

        lmp_tabletop_ui = setup_LMP(env, cfg_tabletop)
        img_first =env.get_camera_image()
        img_first = Image.fromarray(img_first)
        img_first.save('./temp/scene.png')
        time.sleep(1)
        # logger.scene_image_append(goal_1)
        env.cache_video = []
        try:
            lmp_tabletop_ui(goal_1, f'objects = {env.object_list}')
        except: 
            print('error')
            logger.append_fail()
            continue
        gui.display_goal(goal_1, preference)
        gui.root.update()
        rendered_clip = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
        stop_tick = gui.display_video(rendered_clip)
        if stop_tick == -1: flag = False; prev = False
        else: 
            flag = True
            if stop_tick > 50: 
                before_tick = stop_tick - 20
                previous_img = env.cache_video[before_tick]
                previous_img = Image.fromarray(previous_img)
                previous_img.save('./temp/previous.png')
                prev = True
            else: 
                prev = False
        last_img = env.cache_video[stop_tick]
        last_img = Image.fromarray(last_img)
        last_img.save('./temp/stop.png')
        if gui.label ==3:
            continue
        time.sleep(3)
        reason = logger.user_feedback_append(flag=flag, goal = goal_1, prev = prev)
        logger.append_feedback(reason)
        num_iter += 1


    obj_list = logger.scene_generation()
    logger.generate_goal_prompt(obj_list)

    _ = env.reset(obj_list)
    lmp_tabletop_ui = setup_LMP(env, cfg_tabletop)
    img_first =env.get_camera_image()
    img_first = Image.fromarray(img_first)
    img_first.save('./temp/scene.png')
    time.sleep(1)
    # logger.scene_image_append()

    goal_1, preference = logger.goal_generation()
    goal_1 = goal_1.split("=")[-1]
    preference = preference.split("=")[-1]

    env.cache_video = []
    lmp_tabletop_ui(goal_1, f'objects = {env.object_list}')

    gui.display_goal(goal_1, preference)
    gui.root.update()
    rendered_clip = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
    stop_tick = gui.display_video(rendered_clip)
    if stop_tick == -1: flag = False
    else: 
        flag = True
    if not flag: name = '{}_true'.format(args.name)
    else: name = '{}_false'.format(args.name)

    #save video
    rendered_clip.write_videofile(f'./temp/{name}.mp4')


    #save log
    with open(f'./temp/{name}.txt', 'w') as f:
        for line in logger.messages:
            f.write(str(line)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=int, default=0)
    args = parser.parse_args()
    main(args)