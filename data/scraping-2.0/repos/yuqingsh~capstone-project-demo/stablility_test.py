import gurobipy as gp
from gurobipy import GRB
from eventlet.timeout import Timeout

# import auxillary packages
import requests  # for loading the example source code
import openai

# import flaml and autogen
from flaml import autogen
from flaml.autogen.agentchat import Agent, UserProxyAgent
from optiguide.optiguide import OptiGuideAgent

import sys
import numpy as np
import time

np.random.seed(42)


message_set = [
    "What if the throughput of Birmingham increased by 15%?",
    "What will happen if the throughput of Birmingham is increased by 15%?",
    "What would happen if Birmingham's throughput were to increase by 15%?",
    "What would happen if Birmingham's throughput was boosted by fifteen percent?",
    "If Birmingham's throughput is boosted by 15%, what will happen?",
    "What would happen if Birmingham's throughput was to grow by 15%?",
    "What would happen if Birmingham's throughput were to increase by 15%?",
    "What would happen if Birmingham's throughput was boosted by fifteen percent?",
    "How would a 15 percent increase in Birmingham's throughput impact the situation?",
    "What would be the consequences of a 15 percent ise in throughput in Birmingham?",
    "In what ways would Birmingham be affected if its throughput grew by 15%?",
    "What outcomes can be expected if there's a 15 percent escalation in Birmingham's throughput?",
    "How would Birmingham's operations change with a 15 percent enhancement in throughput?",
    "What implications would a 15 percent augmentation in Birmingham's throughput have?",
]

counter = 0

with open("log.txt", "w") as f:
    sys.stdout = f
    sys.stderr = f

    for i in range(50):
        autogen.oai.ChatCompletion.start_logging()

        config_list = autogen.config_list_from_json(
            "OAI_CONFIG_LIST",
            filter_dict={
                "model": {
                    "gpt-4",
                    "gpt4",
                    "gpt-4-32k",
                    "gpt-4-32k-0314",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k",
                    "gpt-3.5-turbo-0301",
                    "chatgpt-35-turbo-0301",
                    "gpt-35-turbo-v0301",
                }
            },
        )
        openai.api_key = "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"

        code = open("supply_network.py", "r").read()  # for local files

        # show the first head and tail of the source code
        print("\n".join(code.split("\n")[:10]))
        print(".\n" * 3)
        print("\n".join(code.split("\n")[-10:]))

        # TODO: add example QA below
        example_qa = """
        ----------
        Question: What if we the throughput of Newcastle is halfed?
        Answer Code:
        ```python
        through["Newcastle"] = through["Newcastle"] / 2

        ```

        ----------
        Question: What if the demand from all our customers all decreased by 5%?
        Answer Code:
        ```python
        demand = demand * 0.95
        ```

        """

        agent = OptiGuideAgent(
            name="Supply Network Example",
            source_code=code,
            debug_times=1,
            example_qa="",
            llm_config={
                "request_timeout": 600,
                "seed": 42,
                "config_list": config_list,
            },
        )

        user = UserProxyAgent(
            "user",
            max_consecutive_auto_reply=0,
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        user.initiate_chat(
            agent, message=message_set[np.random.randint(len(message_set))]
        )

        print("one cycle done, sleep for 60s")
        time.sleep(60)

print(counter)
