import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from argparse import ArgumentParser
import pickle
import minerl
import argparse
import copy
import os
import platform
from queue import Queue
from FindCaveEnvWarper import FindCaveEnvWarper
from utils.torch_utils import select_device, smart_inference_mode
from FindCaveEnvWarper import FindCaveEnvWarper
from DetectCave import DetectCave
from argparse import ArgumentParser
import pickle
import math
import cv2
import numpy as np
from enum import Enum
from findcave.config import STATE, ACTION, IMG_WIDTH, LOCAL_DEDUG
from openai_vpt.agent import MineRLAgent
from findcave.utils.tools import BubbleCheck

def rotateToPos(pos, env):
    rotateAngle = 8 * np.sign(pos / IMG_WIDTH - 0.5) #  (pos / IMG_WIDTH - 0.5) * 51.2 * 2 / 6 # (pos-IMG_WIDTH/2) / IMG_WIDTH * 51.2 * 2 / 6
    if LOCAL_DEDUG:
        print(rotateAngle)
    minerl_action = env.action_space.sample()
    list_key = list(minerl_action.keys())

    value = [np.array(0) for i in range(24)]
    value[3] = np.array([0, rotateAngle])
    for j, kk in enumerate(list_key):
        minerl_action[kk] = value[j]

    return minerl_action


def area_of_rect(leftup, rightdown):
    assert rightdown[0] - leftup[0] > 0
    assert rightdown[1] - leftup[1] > 0

    return (rightdown[0] - leftup[0]) * (rightdown[1] - leftup[1])


def minerl_go_action(game_env):
    go = game_env.action_space.sample()
    list_key = list(go.keys())
    value = [np.array(0) for j in range(24)]
    value[3] = np.array([0, 0])
    for j, kk in enumerate(list_key):
        go[kk] = value[j]
    go['forward'] = np.array(1)
    return go


def minerl_gojump_action(game_env):
    gojump = game_env.action_space.sample()
    list_key = list(gojump.keys())
    value = [np.array(0) for j in range(24)]
    value[3] = np.array([0, 0])
    for j, kk in enumerate(list_key):
        gojump[kk] = value[j]
    gojump['forward'] = np.array(1)
    gojump['jump'] = np.array(1)
    return gojump

def action_mask(action):
    action["ESC"] = 0
    action['drop'] = np.array([0])
    action['inventory'] = np.array([0])
    action['use'] = np.array([0])
    action['sneak'] = np.array([0])
    # minerl_action['attack'] = np.array([0])
    for num in range(1, 10):
        action['hotbar.{}'.format(num)] = np.array([0])
    return action

def minerl_noop_action(game_env):
    noop = game_env.action_space.sample()
    list_key = list(noop.keys())
    value = [np.array(0) for j in range(24)]
    value[3] = np.array([0, 0])
    for j, kk in enumerate(list_key):
        noop[kk] = value[j]
    return noop

def minerl_terminal_action(game_env):
    terminal = game_env.action_space.sample()
    list_key = list(terminal.keys())
    value = [np.array(0) for j in range(24)]
    value[3] = np.array([0, 0])
    for j, kk in enumerate(list_key):
        terminal[kk] = value[j]
    terminal['ESC'] = np.array(1)
    return terminal

def unzip_find_cave_models(path):
    os.system(f"""cd {path} && 7z x findcave.zip.001 -y """)

def main(vpt_model, weights, yolo_weights, env, n_episodes=3, max_steps=int(1e9), show=False, now_i=0):
    # yolo_weights = os.path.join(ROOT, )
    env_name = env
    # remove unzip model phase
    # merely move loop forward to initial model each time and avoid unexcepted exit
    # best diff without whitespace change
    for _ in range(n_episodes):
        if LOCAL_DEDUG:
            if not os.path.exists('debug_sequence' + str(now_i)):
                os.mkdir('debug_sequence' + str(now_i))

        env = FindCaveEnvWarper(env_name)
        device = select_device('cuda:0')
        detect_cave = DetectCave(device,yolo_weights)
        # Using aicrowd_gym is important! Your submission will not work otherwise

        agent_parameters = pickle.load(open(vpt_model, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        agent = MineRLAgent(env.env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
        agent.load_weights(weights)

        # ROTATE_ACTION_LIST = [ACTION.ROTATE for i in range(4)]
        go = minerl_go_action(env)
        gojump = minerl_gojump_action(env)
        noop = minerl_noop_action(env)
        terminal = minerl_terminal_action(env)
        state = STATE.VPT_WALKING
        bubble_check = BubbleCheck('findcave/utils/bubble_template.png', 'findcave/utils/bubble_mask.png')
        previous10_img = Queue(10)
        pre_state = STATE.VPT_WALKING
        done = False
        obs = env.reset()
        step = 0
        for _ in range(max_steps):
            img = obs['pov']
            if LOCAL_DEDUG:
                print(step)
                print("state:", state, pre_state, step)

                cv2.imwrite("debug_sequence"+str(now_i)+"/" + str(step) + ".png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not previous10_img.full():
                previous10_img.put(img)
            else:
                previous10_img.get()
                previous10_img.put(img)
                if step % 100:
                    diff_count = 0
                    if state == STATE.VPT_WALKING: #or state == STATE.GO_TO_CAVE:
                        while not previous10_img.empty():
                            prev = previous10_img.get()
                            difference = cv2.subtract(prev, img)
                            if abs(np.average(difference)) < 2:
                                diff_count += 1
                                if LOCAL_DEDUG:
                                    print("stuckstuckstuckstuckstuckstuckstuckstuck")
                        if diff_count > 2:
                            env.is_stuck = True

                # print(img.shape
            if state == STATE.VPT_WALKING:
                all_pos_set, agent_in_cave = detect_cave.detect_image(img, device, step, now_i)
                if agent_in_cave:
                    obs, reward, done, info = env.step(terminal)
                else:
                    if 'cave' in all_pos_set:
                        pos_set = all_pos_set['cave']
                    elif 'hole' in all_pos_set:
                        pos_set = all_pos_set['hole']
                    else:
                        pos_set = []
                    if len(pos_set) > 0:
                        pre_state = state
                        state = STATE.FIND_CAVE_AND_ROTATE
                        rotate_times = int(abs((pos_set[0][0] + pos_set[0][2])/2 - img.shape[1] / 2) / (320/45) / 8) # every time rotate 8
                        ROTATE_ACTION_IDX = 0
                        obs, reward, done, info = env.step(noop)
                        # continue
                    else:
                        minerl_action = agent.get_action(obs)
                        minerl_action = action_mask(minerl_action)
                        if bubble_check.check(img):
                            minerl_action['jump'] = np.array(1)
                        minerl_action['camera'] = minerl_action['camera'][0]
                        obs, reward, done, info = env.step(minerl_action, img, True)
                        pre_state = state
            elif state == STATE.FIND_CAVE_AND_ROTATE:
                pos = (pos_set[0][0] + pos_set[0][2]) / 2
                if ROTATE_ACTION_IDX == 0:
                    pre_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # if ROTATE_ACTION_IDX > len(ROTATE_ACTION_LIST) or math.fabs(pos - img.shape[1] / 2) < 10:
                if ROTATE_ACTION_IDX >= rotate_times:  # or math.fabs(pos - img.shape[1] / 2) < 8:
                    pre_state = state
                    state = STATE.GO_TO_CAVE
                    ROTATE_ACTION_IDX = 0
                    obs, reward, done, info = env.step(noop)
                else:
                    ROTATE_ACTION_IDX += 1

                    rect_left_up = (pos_set[0][0], pos_set[0][1])
                    rect_right_down = (pos_set[0][2], pos_set[0][3])
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(pre_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    new_rect_left_up = rect_left_up + flow[rect_left_up[1], rect_left_up[0]]
                    new_rect_right_down = rect_right_down + flow[rect_right_down[1] - 1, rect_right_down[0] - 1]
                    rect_left_up = (np.clip(new_rect_left_up[0], 0, flow.shape[1] - 1),
                                    np.clip(new_rect_left_up[1], 0, flow.shape[0] - 1))
                    rect_right_down = (np.clip(new_rect_right_down[0], 0, flow.shape[1] - 1),
                                       np.clip(new_rect_right_down[1], 0, flow.shape[0] - 1))

                    rect_left_up = tuple(np.int32(rect_left_up))
                    rect_right_down = tuple(np.int32(rect_right_down))
                    minerl_action = rotateToPos(pos, env)
                    if LOCAL_DEDUG:
                        show_img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)
                        cv2.rectangle(show_img, rect_left_up, rect_right_down, (0, 0, 255), 3)
                        # cv2.imshow('yolo2', show_img)
                        print("myrotae: ", minerl_action)
                        print("i=", ROTATE_ACTION_IDX)
                        print("action END")
                        # env.render()
                        print("render END")
                    obs, reward, done, info = env.step(minerl_action, img, False)
                    pre_gray = gray
                    # img = obs['pov']
                    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif state == STATE.GO_TO_CAVE:
                # first time enter GOTOCAVE
                if pre_state == STATE.FIND_CAVE_AND_ROTATE:
                    all_pos_set, agent_in_cave = detect_cave.detect_image(img, device, step, now_i)
                    if agent_in_cave:
                        obs, reward, done, info = env.step(terminal)
                    else:
                        if 'cave' in all_pos_set:
                            pos_set = all_pos_set['cave']
                        elif 'hole' in all_pos_set:
                            pos_set = all_pos_set['hole']
                        # else:
                        #     pos_set = []
                        # pos_set from now or vpt state result
                        if len(pos_set) > 0:
                            rect_left_up = (pos_set[0][0], pos_set[0][1])
                            rect_right_down = (pos_set[0][2], pos_set[0][3])
                            center = ((rect_left_up[0] + rect_right_down[0]) / 2, (rect_left_up[1] + rect_right_down[1]) / 2)

                        obs, reward, done, info = env.step(gojump, img)
                        next_action = Queue()

                        if center[1] / img.shape[0] <= 1 / 2:
                            for n in range(2):
                                next_action.put(copy.deepcopy(gojump))
                            for n in range(2):
                                next_action.put(copy.deepcopy(go))

                        elif 1 / 2 < center[1] / img.shape[0] < 3 / 4:
                            # next_action.put(copy.deepcopy(gojump))
                            for n in range(4):
                                next_action.put(copy.deepcopy(go))

                        elif center[1] / img.shape[0] >= 3 / 4:
                            # next_action.put(copy.deepcopy(gojump))
                            for n in range(2):
                                next_action.put(copy.deepcopy(go))

                        pre_state = state
                        state = STATE.GO_TO_CAVE
                        contius_not_found = 0

                elif pre_state == STATE.GO_TO_CAVE:
                    all_pos_set, agent_in_cave = detect_cave.detect_image(img, device, step, now_i)
                    if agent_in_cave:
                        obs, reward, done, info = env.step(terminal)
                    else:
                        if not next_action.empty():  # do macro action
                            now_action = next_action.get()
                            obs, reward, done, info = env.step(now_action, img, False)
                        else:
                            if 'cave' in all_pos_set:
                                cur_pos_set = all_pos_set['cave']
                            elif 'hole' in all_pos_set:
                                cur_pos_set = all_pos_set['hole']
                            else:
                                cur_pos_set = []
                            if len(cur_pos_set) > 0:
                                pre_state = state
                                state = STATE.FIND_CAVE_AND_ROTATE
                                ROTATE_ACTION_IDX = 0
                                contius_not_found = 0
                                pos_set = cur_pos_set
                                rect_left_up = (pos_set[0][0], pos_set[0][1])
                                rect_right_down = (pos_set[0][2], pos_set[0][3])
                                center = (
                                (rect_left_up[0] + rect_right_down[0]) / 2, (rect_left_up[1] + rect_right_down[1]) / 2)
                                rotate_times = int(abs((pos_set[0][0] + pos_set[0][2])/2 - img.shape[1] / 2) / (320/45) / 8)
                            else:
                                contius_not_found += 1
                            # queue always empty
                            if contius_not_found <= 10:
                                # if target lost, rotate to the postion(before doing macro action) slowly
                                # TODO more intelligence, look around?
                                # left: - , down +
                                rect_left = pos_set[0][0]
                                rect_right = pos_set[0][2]
                                rect_down = pos_set[0][3]
                                rotateAngle1 = 0  # (160 - center[1]) / 320 * 51.2 * 2 / 15
                                if LOCAL_DEDUG:
                                    print('rotate:', pos_set, center)
                                if pos_set[0][-1] == 'hole':
                                    if center[1] / 360 > 0.80 or center[0] / 640 < 0.07 or center[0] / 640 > 0.93:
                                    # if center[0] / 640 > 0.80 or center[1]/360 < 0.07 or center[1]/360 > 0.93:
                                        print('missed hole')
                                        if center[0] - 320 > 0:
                                            rotateAngle2 = 30 # (center[0] - 320) / 640 * 51.2 * 2 / 10
                                        else:
                                            rotateAngle2 = -30
                                    else:
                                        rotateAngle2 = 2 * np.sign(center[0] - 320)# (center[0] - 320) / 640 * 51.2 * 2 / 12
                                elif pos_set[0][-1] == 'cave':
                                    rotateAngle2 = 2 * np.sign(center[0] - 320) # (center[0] - 320) / 640 * 51.2 * 2 / 12
                                minerl_action = env.action_space.sample()
                                list_key = list(minerl_action.keys())
                                value = [np.array(0) for i in range(24)]
                                value[3] = np.array([rotateAngle1, rotateAngle2])
                                for j, kk in enumerate(list_key):
                                    minerl_action[kk] = value[j]
                                obs, reward, done, info = env.step(minerl_action, img)
                            else:
                                pre_state = state
                                state = STATE.VPT_WALKING
                                obs, reward, done, info = env.step(noop)
                            if LOCAL_DEDUG:
                                print("contiusnot = ", contius_not_found)

            step += 1
            if done:
                detect_cave.reset()
                state = STATE.VPT_WALKING
                pre_state = STATE.VPT_WALKING
                break

    env.close()


if __name__ == "__main__":
    main(
        vpt_model=r"../findcave/findcave_models/foundation-model-1x.model",
        weights="../findcave/findcave_models/MineRLBasaltFindCave.weights",
        yolo_weights='../findcave/findcave_models/best.pt',
        model_base_dir='../findcave/findcave_models',
        env="MineRLBasaltFindCave-v0",
        n_episodes=1,
        max_steps=3600,
        now_i=999
    )
