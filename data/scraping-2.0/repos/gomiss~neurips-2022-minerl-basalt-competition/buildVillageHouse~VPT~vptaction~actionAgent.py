import copy
import numpy as np
import sys 
sys.path.append("/minerl/basalt-2022-behavioural-cloning-baseline") 
sys.path.append("/minerl/basalt-2022-behavioural-cloning-baseline/action") 

from openai_vpt.lib.actions import ActionTransformer

import torch as th
from actionMap import CameraHierarchicalMappingV2, CameraHierarchicalMappingMoving
from openai_vpt.lib.action_mapping import CameraHierarchicalMapping
import json

# Template action
ENV_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}
QUEUE_TIMEOUT = 10

# Mapping from JSON keyboard buttons to MineRL actions
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)


class ActionAgent:
    """

    转换各个类型(json, env, agent, factor_agent)的动作数据

    __init__:
        env2agent_action_map: CameraHierarchicalMappingV2

    """
    def __init__(self, env2agent_action_map = CameraHierarchicalMappingMoving(n_camera_bins=11)) -> None:
        self.env_action_template = ENV_ACTION.copy()
        self.CAMERA_SCALER = 360.0 / 2400.0
        
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
        self.env2agent_action_map = env2agent_action_map

    def json_action_file_data_loader_preprocess(self, json_path):
        env_action_list = []
        attack_is_stuck = False
        last_hotbar = 0
        with open(json_path) as json_file:
            json_lines = json_file.readlines()
            json_data = "[" + ",".join(json_lines) + "]"
            json_data = json.loads(json_data)
        for i in range(len(json_data)):
            step_data = json_data[i]
            if i == 0:
                # Check if attack will be stuck down
                if step_data["mouse"]["newButtons"] == [0]:
                    attack_is_stuck = True
            elif attack_is_stuck:
                # Check if we press attack down, then it might not be stuck
                if 0 in step_data["mouse"]["newButtons"]:
                    attack_is_stuck = False
            # If still stuck, remove the action
            if attack_is_stuck:
                step_data["mouse"]["buttons"] = [button for button in step_data["mouse"]["buttons"] if button != 0]

            action, is_null_action = self.json_action_to_env_action(step_data)

            current_hotbar = step_data["hotbar"]
            if current_hotbar != last_hotbar:
                action["hotbar.{}".format(current_hotbar + 1)] = 1
            last_hotbar = current_hotbar

            # action["hotbar.{}".format(current_hotbar + 1)] = 1
            
            env_action_list.append(action)

        return env_action_list

    

    def json_action_to_env_action(self, json_action):
        """
        Converts a json action into a MineRL action.
        Returns (minerl_action, is_null_action)
        """
        # This might be slow...
        env_action = self.env_action_template.copy()
        # As a safeguard, make camera action again so we do not override anything
        env_action["camera"] = np.array([0.0, 0.0])

        is_null_action = True
        keyboard_keys = json_action["keyboard"]["keys"]
        for key in keyboard_keys:
            # You can have keys that we do not use, so just skip them
            # NOTE in original training code, ESC was removed and replaced with
            #      "inventory" action if GUI was open.
            #      Not doing it here, as BASALT uses ESC to quit the game.
            if key in KEYBOARD_BUTTON_MAPPING:
                env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
                is_null_action = False

        mouse = json_action["mouse"]
        camera_action = env_action["camera"]
        camera_action[0] = mouse["dy"] * self.CAMERA_SCALER
        camera_action[1] = mouse["dx"] * self.CAMERA_SCALER

        if mouse["dx"] != 0 or mouse["dy"] != 0:
            is_null_action = False
        else:
            if abs(camera_action[0]) > 180:
                camera_action[0] = 0
            if abs(camera_action[1]) > 180:
                camera_action[1] = 0

        mouse_buttons = mouse["buttons"]
        if 0 in mouse_buttons:
            env_action["attack"] = 1
            is_null_action = False
        if 1 in mouse_buttons:
            env_action["use"] = 1
            is_null_action = False
        if 2 in mouse_buttons:
            env_action["pickItem"] = 1
            is_null_action = False

        return env_action, is_null_action

    def env_action_to_agent_action(self, env_action):
        # print("!!!!!!:",env_action)
        # env_action['camera'] = np.array(env_action['camera'])
        policy_action = self.action_transformer.env2policy(env_action)
        if policy_action["camera"].ndim == 1:
            policy_action = {k: v[None] for k, v in policy_action.items()}
    
        # print(policy_action['buttons'])
        
        # if sum(policy_action['buttons'][0])>1:
        #     print(1)

        agent_action = self.env2agent_action_map.from_factored(policy_action)

        return agent_action

    def json_action_to_agent_action(self, json_action):
        env_action, isnull = self.json_action_to_env_action(json_action)
        print("env_action:", env_action)
        agent_action = self.env_action_to_agent_action(env_action)
        return agent_action

    def agent_action_to_env_action(self, agent_action):
        action = agent_action
        if isinstance(action["buttons"], th.Tensor):
            action = {
                "buttons": agent_action["buttons"].cpu().numpy(),
                "camera": agent_action["camera"].cpu().numpy()
            }
        minerl_action = self.env2agent_action_map.to_factored(action)
        env_action = self.action_transformer.policy2env(minerl_action)

        return env_action
    def agent_action_to_json_actiion(self, agent_action):
        pass
    
    def _is_null_json_action(self, json_action):
        pass

    def _is_null_env_action(self, env_action):
        pass

    def _is_null_agent_aciton(self, agent_action):
        pass



if __name__=="__main__":
    actionAgent = ActionAgent()
    json_action = {
    "mouse": {
        "x": -539.0, 
        "y": 898.0, 
        "dx": 0.0, 
        "dy": -1.0, 
        "scaledX": -1179.0, 
        "scaledY": 538.0, 
        "dwheel": 0.0, 
        "buttons": [0], 
        "newButtons": [0]
        }, 
    "keyboard": {
        "keys": [], 
        "newKeys": [], 
        "chars": " "
        }, 
    "hotbar": 0, "tick": 796, "isGuiOpen": False
    }
    # print("json_action:",json_action)
    agent_action = actionAgent.json_action_to_agent_action(json_action)
    print("agent_action:", agent_action)

    # json_path = "/data/MineRLBasaltBuildVillageHouse-v0/cheeky-cornflower-setter-0a9ad3ddd136-20220726-193610.jsonl"
    # env_action_list = actionAgent.json_action_file_data_loader_preprocess(json_path)

    # for env_action in env_action_list:
    #     env_action['camera'] = np.array(env_action['camera'])
    #     agent_action = actionAgent.env_action_to_agent_action(env_action)
        
    # env_action = env_action_list[0]
    # env_action['jump'] = 1
    # env_action['use'] = 1
    # print(env_action)

    # env_action['camera'] = np.array(env_action['camera'])
    # agent_action = actionAgent.env_action_to_agent_action(env_action)
    # print(agent_action)