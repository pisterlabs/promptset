import os
from abc import ABC, abstractmethod
from functools import partial

import openai
import requests
from bardapi import Bard


def compose_prompt_skeleton(num_actions, state_size, first_prompt=False):
    action_space = r"<" + ",".join([str(i) for i in range(num_actions)]) + r">"

    prompt_instructions = open("prompt/instruction.txt", "r").read().format(state_space_length=state_size,
                                                                             action_space=action_space)
    prompt_answer_format = open("prompt/answer_format.txt", "r").read()
    prompt_highlights = open("prompt/highlights.txt", "r").read()

    if first_prompt:
        prompt_skeleton = open("prompt/first_prompt.txt", "r").read()
    else:
        prompt_skeleton = open("prompt/prompt.txt", "r").read()

    prompt_skeleton = prompt_skeleton.replace("{instruction}", prompt_instructions).replace(
        "{answer_format_instructions}", prompt_answer_format).replace("{highlights}", prompt_highlights)

    return prompt_skeleton


class LLMAgent:
    def __init__(self, num_actions, state_size):
        self.first_prompt = compose_prompt_skeleton(num_actions, state_size, first_prompt=True)
        self.prompt = compose_prompt_skeleton(num_actions, state_size, first_prompt=False)

        self.num_actions = num_actions

        self.curr_model = ""
        self.curr_policy = ""
        self.prev_state = ""
        self.last_action = ""
        self.curr_state = ""
        self.curr_reward = 0

        # create file to store model responses, remove old one if exist:
        self.model_responses_file = "last_conversation.txt"
        if os.path.exists(self.model_responses_file):
            os.remove(self.model_responses_file)
        open(self.model_responses_file, "w").close()

        # TODO: add history
        # self.history =

    def reset_agent(self, state):
        self.curr_state = state
        self.prev_state = ""
        self.curr_reward = 0

        prompt = self.first_prompt.format(state=state)
        with open(self.model_responses_file, "a") as f:
            f.write("\nprompt:\n ---------- \n" + prompt + "\n----------------\n")

        llm_response = self.prompt_model_for_next_step(prompt)
        with open(self.model_responses_file, "a") as f:
            f.write("\nresponse:\n ---------- \n" + llm_response + "\n----------------\n")
        action, model, policy = self.parse_llm_response(llm_response)

        self.curr_model = model
        self.curr_policy = policy
        self.last_action = action

        return action

    def step(self, state, reward):
        self.prev_state = self.curr_state
        self.curr_state = state
        self.curr_reward = reward

        next_prompt = self.prompt.format(previous_state=self.prev_state, previous_action=self.last_action, state=state,
                                         reward=reward, model=self.curr_model, policy=self.curr_policy,)
        with open(self.model_responses_file, "a") as f:
            f.write("\nprompt:\n ---------- \n" + next_prompt + "\n----------------\n")

        llm_response = self.prompt_model_for_next_step(next_prompt)
        with open(self.model_responses_file, "a") as f:
            f.write("\nresponse:\n ---------- \n" + llm_response + "\n----------------\n")
        action, model, policy = self.parse_llm_response(llm_response)
        self.curr_model = model
        self.curr_policy = policy
        self.last_action = action

        print("state: ", state)
        print("reward: ", reward)
        print("model: ", self.curr_model)
        print("policy: ", self.curr_policy)
        print("action: ", action)
        print("---------------------")

        return action

    def parse_llm_response(self, llm_response):
        model = llm_response.split("<world model>")[1].split("</world model>")[0]
        policy = llm_response.split("<policy>")[1].split("</policy>")[0]
        # remove line breaks from action
        action = int(llm_response.split("<action>")[1].split("</action>")[0].replace("\n", ""))

        return action, model, policy

    @abstractmethod
    def prompt_model_for_next_step(self, next_prompt):
        pass


# noinspection PyUnreachableCode
class BardAgent(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.session = requests.Session()
        # TODO: remove token from here!
        assert False, "add bard token to LLMAgents.py!"
        self.bardbard = Bard(token="",
                             session=self.session)

    def prompt_model_for_next_step(self, next_prompt):
        return self.bardbard.get_answer(next_prompt)['content']


class ChatGPT35Agent(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        openai.api_key = os.getenv("OPENAI_CRS_KEY")

        self.tokens_used = 0

    def prompt_model_for_next_step(self, next_prompt):
        messages = [{"role": "user", "content": next_prompt},]
        res = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

        tokens_used = res["usage"]["total_tokens"]
        self.tokens_used += tokens_used

        response = res["choices"][0]["message"]["content"]

        return response


class Davinci3Agent(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        openai.api_key = os.getenv("OPENAI_CRS_KEY")

        self.tokens_used = 0

    def prompt_model_for_next_step(self, next_prompt):
        # messages = [{"role": "user", "content": next_prompt},]
        res = openai.Completion.create(model="text-davinci-003", prompt=next_prompt, max_tokens=2000)

        tokens_used = res["usage"]["total_tokens"]
        self.tokens_used += tokens_used

        response = res["choices"][0]["text"]

        return response