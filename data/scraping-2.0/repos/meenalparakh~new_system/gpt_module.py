import openai

OPENAI_KEY = "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"
openai.api_key = OPENAI_KEY

import os
import pickle
import json
from colorama import Fore
import time

RESPONSE_DIR = "gpt_responses"


class ChatGPTModule:
    def __init__(self, response_dir=RESPONSE_DIR, mode="online"):
        """
        allowed mode vals: "human", "online"
        """
        self.response_dir = response_dir
        self.mode = mode
        os.makedirs(response_dir, exist_ok=True)
        self.timer = 31
        self.start_time = 0
        self.time_limit = 30

    def save_cache(self, fname):
        with open(fname, "w") as f:
            json.dump(self.cache, f)

    def start_session(self, session_title):
        session_fname = os.path.join(self.response_dir, f"{session_title}.json")
        self.session_fname = session_fname
        self.message_lst = []

        if os.path.exists(session_fname):
            with open(session_fname, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
            self.save_cache(session_fname)

    def chat(self, prompt, context=None):
        if context:
            self.message_lst.append({"role": "system", "content": context})
            print(Fore.BLUE + "System: " + context)
            print(Fore.BLACK)

        if prompt:
            self.message_lst.append({"role": "user", "content": prompt})
            print(Fore.RED + "User: " + prompt)
            print(Fore.BLACK)

        concatenated_message = "\n".join(
            [m["role"] + ": " + m["content"] for m in self.message_lst]
        )
        # print(concatenated_message)

        if concatenated_message in self.cache:
            print(Fore.MAGENTA + "Found in cache ...")
            result = self.cache[concatenated_message]
        else:
            while self.timer < self.time_limit:
                self.timer = time.time() - self.start_time

            if self.mode == "online":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=self.message_lst
                )
            else:
                human_response_file = input("fname:")

                with open(human_response_file, "r") as f:
                    human_response = f.readlines()

                human_response = "".join(human_response)
                response = {
                    "choices": [
                        {
                            "finish_reason": "done",
                            "message": {"content": human_response},
                        }
                    ]
                }

            self.start_time = time.time()
            self.timer = 0

            finish_reason = response["choices"][0]["finish_reason"]
            if finish_reason != "stop":
                print("warning: the response is truncated")

            result = response["choices"][0]["message"]["content"]
            self.cache[concatenated_message] = result
            self.save_cache(self.session_fname)

        self.message_lst.append({"role": "assistant", "content": result})
        print(Fore.GREEN + "AI: " + result)
        print(Fore.BLACK)
        return result


if __name__ == "__main__":
    # //// Example ////////////////////////////////////////////////////////////////////////////////////////
    chat_module = ChatGPTModule()
    chat_module.start_session("test_run_loop")

    ## // Example chats ///////////////////////////////////////////////////////////////////////////////////////
    # chat_module.chat(prompt="How to place a bowl containing food into a dishwasher?", context="Your job is to tell how to accomplish some task.")
    # chat_module.chat(prompt="Nice job! Next tell me where to place the bowl after it has been washed. I see a bin and a rack.")
    # chat_module.chat(prompt="write code for drawing a circle on a paper. You can use numpy and math libraries. The function you should make use of is `move(position)`, where `position` is a 2d array and moves the pencil to `position` on the paper.")
    # ////////////////////////////////////////////////////////////////////////////////////////////////////

    ## // Example with planner ///////////////////////////////////////////////////////////////////////////////////////
    # from prompt_manager import get_plan, execute_plan

    # primitives_lst = "`find(object)`,`pick_bowl`, `place_bowl`"
    # primitives_description = """
    # ```
    # def find(object):
    # 	# finds the relevant objects in the environment
    # 	# returns the [x, y, z] location of the object.

    # def pick(position):
    # 	#  picks up the bowl located at position
    # 	# position: [x, y, z] coordinate of the bowl

    # def place(position):
    # 	#  places the bowl at a location given by position, [x, y, z]
    # 	# position: [x, y, z] coordinate for desired place location

    # def learn_skill(skill_name, skill_inputs):
    # 	# Arguments:
    # 	# skill_name: a short name for the skill to learn
    # 	# skill_inputs: the arguments that the new skill to be learned should take

    # 	# Returns:
    # 	# the function skill_name that can perform the skill given by “skill_name” and takes as input the skill_inputs.
    # ```
    #     """

    # task_prompt = "place the mug into the tray"
    # description = "On the table lies a tray and a mug. At the right of all the objects on the table lies the tray. Further away to the left of the tray, the mug has been placed."
    # task_name, code_rectified = get_plan(description, task_prompt, chat_module.chat, "place_mug", primitives_lst, primitives_description)

    # print(" FINAL ANSWER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(task_name)
    # print(code_rectified)
    # ////////////////////////////////////////////////////////////////////////////////////////////////////

    # // Example with planner - (loop) ///////////////////////////////////////////////////////////////////////////////////////
    from prompt_manager import get_plan_loop

    primitives_lst = "`find(object)`,`pick_bowl`, `place_bowl`"
    primitives_description = """
```
def find(object):
	# finds the relevant objects in the environment
	# returns the [x, y, z] location of the object.

def pick(position):
	#  picks up the bowl located at position
	# position: [x, y, z] coordinate of the bowl

def place(position):
	#  places the bowl at a location given by position, [x, y, z]
	# position: [x, y, z] coordinate for desired place location

```
    """
    descriptions = [
        "On the table lies a tray and a mug. At the right of all the objects on the table lies the tray. Further away to the left of the tray, the mug has been placed.",
        "The tray contains the mug.",
        "The tray contains the mug.",
        "To the right is the mug, and the tray is placed over it.",
    ]

    task_prompts = [
        "Place the mug into the tray.",
        "Now tell me where the mug lies?",
        "Put the tray over the mug.",
        "What all the objects are present in the scene?",
    ]
    verbal_queries = ["N", "Y", "N", "Y"]

    task_names = ["mug_inside_tray", "", "tray_over_mug", ""]

    ##### interactions

    first_code_run = True

    prompt_idx = 0
    N = len(task_prompts)
    while prompt_idx < N:
        sd = descriptions[prompt_idx]
        tp = task_prompts[prompt_idx]
        tn = task_names[prompt_idx]

        is_verbal = verbal_queries[prompt_idx] == "Y"

        response = get_plan_loop(
            sd,
            tp,
            chat_module.chat,
            tn,
            primitives_lst,
            primitives_description,
            code_rectification=first_code_run,
            first_run=first_code_run,
            verbal_query=is_verbal,
        )

        if is_verbal:
            print(
                " FINAL ANSWER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            print(response)
        else:
            print(
                " FINAL ANSWER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            )
            task_name, code_str = response
            print(task_name)
            print(code_str)

        if not is_verbal:
            first_code_run = False

        prompt_idx += 1
