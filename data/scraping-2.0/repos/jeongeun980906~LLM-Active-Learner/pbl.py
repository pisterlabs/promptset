from utils.env import *
from utils.lmp import *
from utils.lmp_wrapper import *
import copy
from utils.cap_prompts import *
import matplotlib.pyplot as plt
from utils.config import *
from moviepy.editor import ImageSequenceClip
import openai
from llm.gui import GUI
from llm.baseline import Baseline
import openai
from utils.key_register import set_openai_api_key_from_txt
import argparse

from llm.baseline import Baseline
from llm.gui import GUI
set_openai_api_key_from_txt(key_path='./key/key.txt',VERBOSE=True)

#@title Initialize Env { vertical-output: true }
high_resolution = False #@param {type:"boolean"}
high_frame_rate = False #@param {type:"boolean"}

def main(args):
# setup env and LMP
    env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
    
    logger = Baseline()


    gui = GUI()
    gui.root.update()
    num_iter = 0
    while True:
        try:
            obj_list = logger.scene_generation()
            inp = input('continue?')
            if inp == 'n': num_iter+= 1;gui.label = 3; continue
            break
        except: pass
    while num_iter < 5:
        if gui.label ==4:
            while True:
                try:
                    obj_list = logger.scene_generation()
                    inp = input('continue?')
                    if inp == 'n': num_iter+= 1;gui.label = 4; continue
                    break
                except: pass
        _ = env.reset(obj_list)
        lmp_tabletop_ui = setup_LMP(env, cfg_tabletop)
        # logger.generate_goal_prompt(obj_list)
        logger.prompt_generation(obj_list)
        goal_1, goal_2, preference = logger.goal_generation(gui.label == 4)
        
        if 'done' in goal_1 or 'done' in goal_2:
            break
        goal_1 = goal_1.split("=")[-1]
        goal_2 = goal_2.split("=")[-1]
        preference = preference.split("=")[-1]

        gui.display_goals(goal_1, goal_2, preference)

        env.cache_video = []
        try:
            lmp_tabletop_ui(goal_1, f'objects = {env.object_list}')
            # display env
            video_1 = env.cache_video
        except:
            video_1 = [env.get_camera_image()]
        print(len(video_1))
        _ = env.reset(obj_list)
        env.cache_video = []
        try:
            lmp_tabletop_ui(goal_2, f'objects = {env.object_list}')
            video_2 = env.cache_video
        except:
            print("error")
            video_2 = [env.get_camera_image()]
        print(len(video_2))
        # display env
        gui.display_videos(video_1, video_2)
        while gui.label == None:
            gui.root.update()
        logger.answer_generation(gui.label)
        num_iter += 1


    obj_list = logger.scene_generation()
    _ = env.reset(obj_list)
    #@title Interactive Demo { vertical-output: true }
    goal, preference = logger.final_goal(obj_list)

    # run policy
    _ = env.reset(obj_list)
    user_input = goal

    env.cache_video = []
    gui.display_goals("final", goal, preference)

    print('Running policy and recording video...')
    lmp_tabletop_ui(user_input, f'objects = {env.object_list}')
    video = env.cache_video
    gui.display_videos(video, [np.zeros_like(video[0])])
    video = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
    # display env
    

    while gui.label == None:
        gui.root.update()
    if gui.label == 1 or gui.label == 2:
        name = "{}_pb_true".format(args.name)
    else: 
        name = "{}_pb_false".format(args.name)
    # import copy
    # render video
    if env.cache_video:
        rendered_clip = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
    #   display(rendered_clip.ipython_display(autoplay=1, loop=1))


    #save video
    rendered_clip.write_videofile(f'./temp/{name}_pbl.mp4')


    #save log
    with open(f'./temp/{name}_pbl.txt', 'w') as f:
        for line in logger.log:
            f.write(line+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=int, default=0)
    args = parser.parse_args()
    main(args)