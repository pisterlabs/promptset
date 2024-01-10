from argparse import ArgumentParser
import pickle
import os,sys
sys.path.append(os.path.dirname(__file__))
import aicrowd_gym
import minerl
import cv2 as cv
import time
import numpy as np
from rule.utils import *
from multiprocessing import Process
import torch as th
from torchvision import datasets, transforms
import torchvision
os.environ['TORCH_HOME'] = os.path.dirname(os.path.dirname(__file__))
from openai_vptV2.agent import MineRLAgent,MineRLAgentMoving

def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=True):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(max_steps):
            action = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)
            if show:
                env.render()
            if done:
                break
    env.close()

class ToWardsState:
    def __init__(self) -> None:

        self.YMAX = 180
        self.XMAX = 360

        self.x = 90.0
        self.y = 90.0
        
    def mouse_move(self, dx, dy):
        self.x = (self.x + dx) % self.XMAX

        self.y = min(max(0, (self.y + dy)), self.YMAX)

    def reset(self):
        self.x = 90.0
        self.y = 90.0

    def __repr__(self) -> str:
        return f"x: {self.x}, y: {self.y}"

def dataTransforms(frame):
    train_transforms = transforms.Compose([
                                        #    transforms.Resize((224,224)),
                                        transforms.ToTensor(),    
                                        transforms.Resize((224,224)),                            
                                        torchvision.transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
        ),
                                        ])
    return np.array(train_transforms(frame))


def Noinv(model, weights, env, n_episodes=100, max_steps=int(10000), show=False, no_action_clip=False):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)
    towardstate = ToWardsState()
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    no_action_clip = no_action_clip
    if no_action_clip:
        agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)  
    else:
        agent = MineRLAgentMoving(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)
    flatten_area_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'train','flattenareaV5mobilenet.pt')
    flatten_area_model = th.load(flatten_area_model_path)
    succeed_count = 0

    episode_i = 0
    while episode_i<n_episodes:
        # print("!!!!!!!!!!!!!!!!!!!!")
        # print(f"episode:{episode_i}") 
        hashouse = False
        hasdoor = False
        episode_i+=1
        try:       
            obs = env.reset()
            towardstate.reset()
            last_3_action = [None, None, None]
            last_5_obs = [None, None, None, None, None]
            last_10_obs = [None for _ in range(10)]
            check_flag = True
            true_step = 10

        except:
            env = aicrowd_gym.make(env)
            towardstate = ToWardsState()
            agent_parameters = pickle.load(open(model, "rb"))
            policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
            pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
            pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
            agent = MineRLAgentMoving(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
            agent.load_weights(weights)
            episode_i -= 1
            continue
        for cur_step in range(max_steps):
            # print(cur_step)
            try:
                if cur_step==0:
                    check_style(env,towardstate)
                    continue
                action, agent_action = agent.get_action(obs)
                # print(f"action:{action}")
                # ESC is not :part of the predictions model.
                # For baselines, we just set it to zero.
                # We leave proper execution as an exercise for the participants :)
                action["ESC"] = 0

                realaction = agent.action_mapper.BUTTONS_IDX_TO_COMBINATION[int(agent_action["buttons"].squeeze(1))]

                if "back" in realaction:
                    continue
                if no_action_clip:
                    for sub_ac in ['back', 'drop', 'hotbar.1','hotbar.2','hotbar.3','hotbar.4','hotbar.5','hotbar.6','hotbar.7','hotbar.8','hotbar.9','inventory','swapHands','use']:
                        action[sub_ac] = 0
                if "jump" in realaction:
                    last_3_action = last_3_action[1:] + ['jump']
                else:
                    last_3_action = last_3_action[1:] + ['None']


                if towardstate.y>130:
                    action['camera'][0][0] = -abs(action['camera'][0][0])
                elif towardstate.y<50:
                    action['camera'][0][0] = abs(action['camera'][0][0])


                r,g,b = cal_rgb(obs['pov'])
                if g>50 and b<30:
                    action['attack'] = 1
                if hashouse: action["ESC"] = 1
                obs, _, done, _ = env.step(action)
                if done:
                    break



                last_5_obs = last_5_obs[1:] + [obs['pov']]
                last_10_obs = last_10_obs[1:] + [obs['pov']]
                
                dy, dx = action['camera'][0]

                towardstate.mouse_move(dx,dy)
                if show:
                    env.render()
                if cur_step > true_step:
                    check_flag=True


                is_flatten_area = th.sigmoid(flatten_area_model(th.unsqueeze(th.tensor(dataTransforms(obs['pov'].copy())), dim=0).to('cuda')))

                if is_flatten_area>0.7:
                    vaild_space = True
                else:
                    vaild_space = False

                if cur_step%100==0 :
                    if cur_step>10:
                        is_stuck(env, towardstate, last_10_obs)

                if vaild_space and check_flag:
                    if cur_step<200:
                        continue
                    valid = 1
                    is_water, RGB = eval("turn_horizontal")(env, towardstate)
                    if is_water:
                        print("water!!!!!!!!!!!!", RGB)
                        obs = get_cur_obs(env,1)
                        continue
                    
                    if cur_step>0:
                        if not hasdoor:
                            hashouse, hasdoor = baodi(env,towardstate)
                            if hashouse:
                                succeed_count += 1
                                break
                        elif hasdoor:
                            hashouse, hasdoor = baodiV2(env,towardstate)
                            if hashouse:
                                succeed_count+= 1
                                break
                        continue
                    is_flatten = eval("go_circle")(env,towardstate)
                    if int(valid)==1 and is_flatten:
                        eval("turn")(env, towardstate, [90,90])
                        build_finish = eval("buildVillageHouse")(env, towardstate)
                        if build_finish:
                            succeed_count += 1
                            break
                        else:
                            continue

                    elif not is_flatten:
                        valid = 0
                        check_flag = False
                        true_step = cur_step + 100


                    else:
                        valid = 0
                        continue

                if done:
                    break
            
            except Exception as e:
                print(e)
                break

        print(f"episode_i:{episode_i}, succe_count:{succeed_count}, step:{cur_step}")
    env.close()

def unzipflattenmodel(base_dir):
    hub_dir = os.path.join(base_dir, "hub")
    ckpt_dir = os.path.join(hub_dir, "checkpoints")
    model_dir = os.path.join(base_dir, "models")
    # print(f"hub dir: {hub_dir}, ckpt dir: {ckpt_dir}, model dir: {model_dir}")
    os.system(f"""cd {hub_dir} && 7z x myfiles.zip.001 -y && mv {os.path.join(ckpt_dir,"flattenareaV4resnet50.pt")} {model_dir}  && mv {os.path.join(ckpt_dir,"MineRLBasaltMovingNoinv.pt")} {model_dir}""")

def unzipmodels():
    base_dir = os.path.dirname(__file__)
    # print(base_dir)
    if not os.path.exists(os.path.join(base_dir, 'models', 'flattenareaV4resnet50.pt')):
        # print(base_dir)
        unzipflattenmodel(base_dir)

    if not os.path.exists(os.path.join(base_dir, 'models', 'MineRLBasaltMovingNoinv.weights')):
        # print(base_dir)
        unzipflattenmodel(base_dir) 

if __name__ == "__main__":
    unzipmodels()

    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    Noinv(args.model, args.weights, args.env, show=args.show)

    # Noinv(
    #     r"E:\MineRL\dataset\basalt-2022-behavioural-cloning-baseline\data\VPT-models\foundation-model-1x.model",
    #     r"E:\MineRL\dataset\basalt-2022-behavioural-cloning-baseline\data\VPT-models\MineRLBasaltMovingNoinv.weights",
    #     "MineRLBasaltBuildVillageHouse-v0", 
    #     show=True)

