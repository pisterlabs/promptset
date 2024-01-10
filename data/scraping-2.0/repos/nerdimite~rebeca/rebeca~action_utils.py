import torch
import numpy as np
from openai_vpt.lib.action_mapping import CameraHierarchicalMapping
from openai_vpt.lib.actions import ActionTransformer

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

# Mapping from JSON keyboard buttons to MineRL actions
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

# Template action
NOOP_ACTION = {
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

CAMERA_SCALER = 360.0 / 2400.0


class ActionProcessor:
    def __init__(self) -> None:

        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    def _json_action_to_env_action(self, json_action):
        """
        Converts a json action into a MineRL action.
        Returns (minerl_action, is_null_action)
        """
        # This might be slow...
        env_action = NOOP_ACTION.copy()
        # As a safeguard, make camera action again so we do not override anything
        env_action["camera"] = np.array([0, 0])

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
        camera_action[0] = mouse["dy"] * CAMERA_SCALER
        camera_action[1] = mouse["dx"] * CAMERA_SCALER

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

    def _env_action_to_agent(
        self,
        minerl_action_transformed,
        to_torch=False,
        check_if_null=False,
        device="cpu",
    ):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(
                minerl_action["camera"] == self.action_transformer.camera_zero_bin
            ):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: torch.from_numpy(v).to(device) for k, v in action.items()}
        return action

    def json_to_action_vector(self, json_actions, to_list=True):
        """
        Converts a list of original json actions into encoded actions.
        """
        buttons = []
        camera = []
        for json_action in json_actions:
            env_action, is_null_action = self._json_action_to_env_action(json_action)
            if not is_null_action:
                agent_action = self._env_action_to_agent(env_action)
                buttons.append(agent_action["buttons"].squeeze().tolist())
                camera.append(agent_action["camera"].squeeze().tolist())

        # convert to one-hot
        # buttons = np.array(buttons)
        # camera = np.array(camera)

        # buttons_vector = np.zeros((buttons.shape[0], 8641))
        # camera_vector = np.zeros((camera.shape[0], 121))

        # for i in range(buttons.shape[0]):
        #     buttons_vector[i, buttons[i]] = 1

        # for i in range(camera.shape[0]):
        #     camera_vector[i, camera[i]] = 1

        # if to_list:
        #     buttons_vector = buttons_vector.tolist()
        #     camera_vector = camera_vector.tolist()

        return {"buttons": buttons, "camera": camera}


def cache_process(situation, state):
    return [], []