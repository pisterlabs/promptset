from pathlib import Path
from loguru import logger
import os
import uuid
import random
import openai
import numpy as np
import imageio
import json
import requests


SKILLS = {"ask_human": 0, "knock_on": 1, "touch": 2, "pick_up": 3, "weigh": 4}
# z and quatenion
DEFAULT_POSE = np.array(
    [0.83443451, 0.76801813, -0.61692601, -0.13173638, 0.11043578, 4]
)  # the last "4" is for release


class Client:
    def __init__(self) -> None:
        self.headers = {"content-type": "application/json"}
        self.address = None

    def call(self, **kwargs):
        data = json.dumps(kwargs)
        response = requests.post(self.address, data=data, headers=self.headers)
        result = json.loads(response.text)
        return result


class ViLDClient(Client):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.address = "http://0.0.0.0:8848/api/vild"
        self.CATEGORY_NAMES = [
            "red block",
            "green block",
            "blue block",
            "orange block",
            "yellow block",
            "purple block",
        ]

    def call(self, category_names=None, **kwargs):
        if category_names is None:
            category_names = self.CATEGORY_NAMES
        return super().call(category_names=category_names, **kwargs)


class SoundClient(Client):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.address = "http://0.0.0.0:8849/api/sound"


class Assistant:
    def __init__(
        self, switch_vision=True, swith_sound=True, sound_use_adjective=True
    ) -> None:
        if swith_sound:
            self.sounder = SoundClient()
            self.sound_use_adjective = sound_use_adjective
        if switch_vision:
            self.vilder = ViLDClient()

    def vision(self, image_path=None, plot_on=False):
        found_objects = self.vilder.call(image_path=image_path, plot_on=plot_on)
        objects = ", ".join([k for k in found_objects])
        caption = f"[{objects}]"
        self.caption = caption
        self.found_objects = found_objects
        logger.debug(self.found_objects)
        return caption

    def sound(self, sounds):
        if len(sounds) == 0:
            sound_like = (
                f"The robot is currently not able to knock on the targeted object."
            )
            return sound_like
        top_probs, top_materials, top_adjectives = self.sounder.call(
            sound_path=sounds[0]
        )
        prob0, material0, adj0 = (
            int(top_probs[0] * 100),
            top_materials[0],
            top_adjectives[0],
        )
        prob1, material1, adj1 = (
            int(top_probs[1] * 100),
            top_materials[1],
            top_adjectives[1],
        )
        if self.sound_use_adjective:
            if prob0 > 50:
                sound_like = f"It sounds {adj0}."  # or f'sounds {sharp}'
            else:
                sound_like = f"It sounds {adj0} mostly and also a little bit {adj1}."  # or f'sounds {sharp}'
        else:
            if prob0 > 85:
                sound_like = f"It is made of {material0}."  # or f'sounds {sharp}'
            elif prob0 > 50:
                sound_like = (
                    f"It is probably made of {material0}."  # or f'sounds {sharp}'
                )
            else:
                sound_like = f"The material cannot be certainly confirmed according to the impact sound. It could be {material0} with a {prob0}% chance, or {material1} with a {prob1}% chance."  # or f'sounds {sharp}'
        logger.info(f"[Sound]: {sound_like}")
        return sound_like

    def touch(self, touchs):
        if len(touchs) > 0:
            feeling = f"It feels {touchs[0]}."
        else:
            feeling = f"Cannot touch it."
        logger.info(f"[Feeling]: {feeling}")
        return feeling

    def weigh(self, weights):
        if len(weights) > 0:
            weighing = f"It weighs {weights[0]}."
        else:
            weighing = f"Not able to weigh it now."
        logger.info(f"[Weight]: {weighing}")
        return weighing

    def feedback(self):
        return

    def target_to_normalized_coordinates(self, target):
        invalid_target = False
        if target not in self.found_objects:
            target = random.choice([k for k in self.found_objects])
            invalid_target = True
        return (
            self.found_objects[target]["normalized_coordinates"],
            target,
            invalid_target,
        )


class Agent:
    def __init__(self, assistant: Assistant, action_tolerance=5) -> None:
        self.actions = {
            "knock_on": self.knock_on,
            "touch": self.touch,
            "weigh": self.weigh,
            "ask_human": self.ask_human,
            "pick_up": self.pick_up,
            "terminate": self.terminate,
        }
        self.assistant = assistant
        self.action_indicator = " robot."
        self.action_tolerance = action_tolerance
        self._clear()

    def reset(self):
        self._clear()

    def execute(self, environment, command):
        # command in a format: "AI: > robot.knock(blue block)"
        if self.action_indicator not in command:
            return "", self.explain(command), "explain", 0, False

        normal = False
        command = command.replace("[", "(").replace("]", ")")
        infos = command.split(self.action_indicator)[1].split("(")
        target, *_explanation = infos[1].split(")")
        if isinstance(_explanation, str):
            explanation = _explanation
        else:
            explanation = "".join(_explanation)
        skill = infos[0].lower()
        if skill not in SKILLS:
            if skill in ["knockon", "knock on", "knock up", "knock_up", r"knock\_on"]:
                skill = "knock_on"
            elif skill in ["pickup", "pick up", "pick on", "pick_up", r"pick\_up"]:
                skill = "pick_up"
            elif skill in ["touchon", "touch_on", "touch up", "touch_up", r"touch\_on"]:
                skill = "touch"
            elif skill in ["weighon", "weigh_up", "weigh up", "weigh_on", r"weigh\_on"]:
                skill = "weigh"
            else:
                skill = "knock_on"
        action, chosen_target, invalid_target = self._target_to_action(target, skill)
        if self.many_duplicates:
            skill = "pick_up"
            reason = "Too many duplicated actions."
        elif self.invalid_count >= self.action_tolerance:
            skill = "pick_up"
            reason = "Too many invalid actions."
        else:
            if invalid_target:
                reason = "Invalid target."
                self.invalid_count += 1
            else:
                normal = True

        skill_func = self.actions[skill]
        logger.debug(f"Carrying out skill {skill_func} on action {action} ...")
        description, explanation, reward, done = skill_func(environment, action)
        if normal:
            return description, explanation, skill, reward, done
        skill_natural = skill.replace("_", " ")
        description_pre = (
            f"Human: {reason} Randomly {skill_natural} the {chosen_target} instead."
        )
        description = description.replace("Human:", "")
        description = description_pre + description
        # count duplicated actions
        if description not in self.executions:
            self.executions[description] = 0
        else:
            self.executions[description] += 1
            if self.executions[description] >= self.action_tolerance:
                self.many_duplicates = True
            else:
                pass
        return description, explanation, skill, reward, done

    def knock_on(self, environment, action):
        action = self._update_action_skill(action, skill="knock_on")
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        sounds = info["sounds"]
        sound_like = self.assistant.sound(sounds)
        description = f"Human: {sound_like}\nAI:"
        return description, None, reward, False or terminate

    def pick_up(self, environment, action):
        env = environment.env
        action = self._update_action_skill(action, skill="pick_up")
        obs, reward, terminate, info = env.step(action)
        description = "Human: Explain why.\nAI:"
        return description, None, reward, True or terminate

    def explain(self, command):
        logger.info(command)
        description = "Human: go on.\nAI:"
        return description, command, 0, False

    def vision(self, *args, **kwargs):
        return self.assistant.vision(*args, **kwargs)

    def touch(self, environment, action):
        action = self._update_action_skill(action, skill="touch")
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        touchs = info["touchs"]
        feeling = self.assistant.touch(touchs)
        description = f"Human: {feeling}\nAI:"
        return description, None, reward, False or terminate

    def ask_human(self, target):
        answer = self.assistant.feedback()
        return answer, None, 0, False

    def weigh(self, environment, action):
        action = self._update_action_skill(action, skill="weigh")
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        weights = info["weights"]
        feeling = self.assistant.weigh(weights)
        description = f"Human: {feeling}\nAI:"
        return description, None, reward, False or terminate

    def terminate(self):
        reward = 0
        return "", None, reward, True

    def _clear(self):
        self.invalid_count = 0
        self.executions = {}
        self.many_duplicates = False

    def _target_to_action(self, target, skill):
        (
            normmalized_coordinates,
            target,
            invalid_target,
        ) = self.assistant.target_to_normalized_coordinates(target)
        return (
            np.r_[normmalized_coordinates, DEFAULT_POSE[:-1], SKILLS[skill]],
            target,
            invalid_target,
        )

    def _update_action_skill(self, action, skill):
        action[-1] = SKILLS[skill]
        return action


class ChatEnvironment:
    def __init__(
        self,
        env_cls,
        mode="test",
        headless=False,
        temp_directory="./temp",
        debug=True,
        render_mode=None,
    ) -> None:
        if render_mode == "None":
            render_mode = None
        env = env_cls(
            observation_mode="vision", headless=headless, render_mode=render_mode
        )
        env.set_mode(mode)  # either 'train' or 'test'
        env.set_random(not debug)
        self.env = env
        self.temp_directory = Path(temp_directory).absolute()
        self.temp_files = []
        self.instruction = None

    def reset(self):
        try:
            obs = self.env.reset()
        except RuntimeError:
            logger.warning("Bad initialization, reset.")
        except Exception as e:
            raise e
        # image = obs["front_rgb"]
        self.instruction = self.env.info["instruction"]
        return

    def instruct(self):
        return self.instruction

    def instruct_with_caption(self, caption=None):
        if caption is None:
            instruction = f'Human: "{self.instruction}".\n'
        else:
            instruction = (
                f'Human: "{self.instruction}" in the scene that contains {caption}.\n'
            )
        return instruction

    def render(self):
        env = self.env
        image = env.states["front_rgb"]
        temp_image_name = str(uuid.uuid4())[-8:] + ".jpg"
        if not self.temp_directory.exists():
            self.temp_directory.mkdir()
        temp_image_path = str(self.temp_directory / temp_image_name)
        imageio.imwrite(temp_image_path, image)
        self.temp_files.append(temp_image_path)
        return temp_image_path

    def clean_up(self):
        for temp_file in self.temp_files:
            os.remove(temp_file)
            logger.warning(f"Removed {temp_file}.")
        self.temp_files = []
        return


LLM_CACHE = {}


class LLM:
    def __init__(
        self,
        engine="Vicuna-13b",
        openai_api_base="",
        openai_api_key="EMPTY",
        prompt_path=None,
        max_tokens=512,
        temperature=0.1,
    ) -> None:
        if len(openai_api_base) > 0:
            openai.api_base = openai_api_base
        openai.api_key = openai_api_key
        if prompt_path is not None:
            with open(prompt_path) as f:
                prompt = f.read() + "\n"
        else:
            prompt = ""
        self._prompt = prompt
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature

    def reset(self):
        pass

    def feed(self, prompt=""):
        if "explain" in prompt.lower():
            max_tokens = 128
        else:
            # For actions, use small tokens.
            max_tokens = 16
        prompt = self._prompt + prompt
        response = self.gpt3_call(prompt, max_tokens=max_tokens)
        command = response["choices"][0]["text"]
        # command = 'robot.knock(blue block)\nother things\n'
        if "\n" in command:
            command = command.split("\n")[0]
        command = command + "\n"
        return command

    def gpt3_call(self, prompt, max_tokens=128, logprobs=1, echo=False):
        max_tokens = min(self.max_tokens, max_tokens)
        temperature = self.temperature
        id = tuple((self.engine, prompt, max_tokens, temperature, logprobs, echo))
        if id in LLM_CACHE.keys():
            logger.warning("cache hit, returning")
            response = LLM_CACHE[id]
        else:
            response = openai.Completion.create(
                # Use `engine` as keyword instead of `model` for the old version of `text-davinci-003`
                model=self.engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                # logprobs=logprobs,
            )
            LLM_CACHE[id] = response
        return response


class FakeLLM(LLM):
    def __init__(self, *args, **kwargs):
        self.count = 0
        self.diversity = [
            "> robot.knock_on(random object)\n",
            "> robot.touch(random object)\n",
            "> robot.weigh(random object)\n",
            "> robot.weigh(random object)\n",
            "> robot.touch(random object)\n",
            "> robot.touch(random object)\n",
            "> robot.touch(random object)\n",
            "> robot.knock_on(random object)\n",
            "> robot.weigh(random object)\n",
            "> robot.knock_on(random object)\n",
        ]

    def reset(self):
        self.count = 0

    def feed(self, *args, **kwargs):
        rtn = self.diversity[self.count]
        self.count += 1
        return rtn
