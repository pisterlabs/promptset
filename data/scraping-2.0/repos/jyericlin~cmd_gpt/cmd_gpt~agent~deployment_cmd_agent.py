# Copyright (c) 2023 Jieyu Lin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
import time

from loguru import logger
from langchain.llms import OpenAI

from cmd_gpt.tool.basic.ask import ask
from cmd_gpt.tool.terminal.iterm2 import iTerm2Interaction, setup_iterm2_tool

EXAMPLE_TEMPLATE = """
Task: Make sure Vim is install in 10.23.12.34
M: terminal_run: 
    ubuntu@labtop:~$
A: I'm on my local machine, and I need to check if Vim is installed on the remote system 10.23.12.34. I can do this by ssh-ing into the remote machine and executing a command to check if Vim is installed.
P: ssh to 10.23.12.34 -> check Vim installation -> if vim not installed, install Vim -> end
E: terminal_run[ssh ubuntu@10.23.12.34]

M: terminal_run: 
    ubuntu@10.23.12.34:~$
A: I'm logged into the remote system. I should check if Vim is installed.
P: check Vim installation -> if vim not installed, execute command to install Vim -> verify installation -> end
E: terminal_run[vim --version]

M: terminal_run: 
    vim: command not found
A: Vim is not installed on this system. I'll need to install it.
P: execute command to install Vim -> verify installation ->  end
E: terminal_run[sudo apt-get install vim]

M: terminal_run: 
    [sudo] password for ubuntu:
A: Require sudo during installation, I need to ask my boss for the password to install Vim.
P: ask password -> enter password -> verify installation -> end
E: ask[Hi boss, I need the password to install Vim on the server]

M: ask reply: 123456
A: I have the password to install Vim. I should enter it into the terminal.
P: enter password -> verify installation -> end
E: terminal_run[123456]

M: terminal_run: 
    ubuntu@10.23.12.34:~$ (After Vim installation is successful)
A: Vim has been installed on the remote system, I should verify the installation.
P: verify installation -> end
E: terminal_run[vim --version]

M: terminal_run:
    VIM - Vi IMproved 8.0 (2016 Sep 12, compiled Apr 10 2018 21:31:58)
    Included patches: 1-1453
    (...omitted 20 lines)
A: Vim has been installed successfully.
P: end
E: end

"""

PROMPT_TEMPLATE = """
Assume you are a deployment engineer agent who is good at deploying software. You goal is to conduct deployment tasks correctly and efficiently to accomplish the deployment goal. You have access to a laptop with terminal and Internet connectivity. You follow the MAPE (Monitor, Analyze, Plan, Execute) thinking cycle:

Monitor: get the observation of what you can see;
Analyze: what should you consider doing based on the observation you have;
Plan: plan your actions
Execute: Take one action or mutiple actions as planed
(The MAPE cycle can loop N time until no action needs to be taken)

The following are the tool you have access to:
1. terminal_run: you can type in the terminal to execute commands and return the last 5 lines of the terminal. Usage: terminal[command]
2. terminal_read: you can read the output of the terminal; it is helpful when you need to read more on the screen. Usage: terminal_read[lines]
3. ask: you can ask your team members or boss if you have question or need help. Usage: ask[content]

Guideline:
1. Do not try to upgrade system unless you absolutely need to
2. DO NOT COPY THE EXAMPLES BELOW!
3. END WHEN EVER YOU CAN!
4. when you are done, write E: end


## Examples
{example}


Task: {task_info}
{agent_scratch_pad}

ONLY PERFORM ONE STEP OF MAPE, make sure A,P,E are all provided. Think if you stop here?
"""



def processing_output(text):
    lines = text.split("\n")

    analysis = ""
    plan = ""
    action = ""


    for line in lines:
        if line[:2] == "M:":
            continue # ignore the monitor part
        elif line[:2] == "A:":
            analysis = line.strip()
        elif line[:2] == "P:":
            plan = line.strip()
        elif line[:2] == "E:":
            action = line.strip()

    if action[2:].strip() in ["end", "exit"]:
        is_done = True
    elif plan[2:].strip() == "end":
        is_done = True
    elif plan == "" and action == "": # no plan an action, assume done
        is_done = True 
    else:    
        is_done = False
    
    
    ret_text = analysis + "\n" + plan + "\n" + action + "\n"

    if not is_done:
        act_parts = action.split("[")
        action = act_parts[0][2:].strip()
        action_input = "[".join(act_parts[1:])[:-1]
    else:
        action = ""
        action_input = ""
    
    return ret_text, action, action_input, is_done

    

class DeploymentCmdAgent():
    def __init__(self, config, llm=None, max_steps=100, action_mapping = {}, cmd_confg={}, max_scrtach_pad_rounds=3):
        
        cmd = setup_iterm2_tool(**cmd_confg)
        self.cmd = cmd
        self.agent_scratch_pad = ""
        self.max_scratch_pad_rounds = max_scrtach_pad_rounds

        if llm is None:
            self.llm = OpenAI(temperature=0, model_name=config["OPENAI_API_MODEL"])
        self.max_steps = max_steps
        if action_mapping == {}:
            self.action_mapping = {
                "terminal_run": cmd.run_command_and_get_reply,
                "terminal_read": cmd.read_output,
                "ask": ask,
            }
    
    def num_scratchpad_rounds(self, scratch_pad):
        count = 0
        for line in scratch_pad.split("\n"):
            if re.match(r"^M: .*: .*", line.strip()):
                count += 1
        return count
    
    def shrink_scratchpad_rounds(self, scratch_pad, n):
        new_scratch_pad = []
        count = 0
        for line in reversed(scratch_pad.split("\n")):
            new_scratch_pad.append(line)
            if re.match(r"^M: .*: .*", line.strip()):
                count += 1
                if count >= n:
                    break
        return "\n".join(list(reversed(new_scratch_pad)))

    def run(self, task_info):
        i = 0
        self.agent_scratch_pad += f"M: terminal_run: " + self.action_mapping["terminal_read"]() + "\n"
        while True:
            self.shrink_scratchpad_rounds(self.agent_scratch_pad, self.max_scratch_pad_rounds)
            if i<2:
                template = PROMPT_TEMPLATE.format(task_info=task_info, agent_scratch_pad=self.agent_scratch_pad, example=EXAMPLE_TEMPLATE)
            else:
                template = PROMPT_TEMPLATE.format(task_info=task_info, agent_scratch_pad=self.agent_scratch_pad, example="")
            logger.debug("="*10 + "template" + "="*10 + "\n" + template + "\n" + "="*10 + "\n")
            output = self.llm.predict(template)
            logger.debug("="*10 + "output" + "="*10 + "\n" + output + "\n" + "="*10 + "\n")
            processed_output, action, action_input, is_done = processing_output(output)
            if action in self.action_mapping:
                logger.info("Running this in 3 seconds: Action: " + str(action) + "; Input: " + str(action_input))
                time.sleep(3)
                obs = self.action_mapping[action](action_input)
                
            else:
                if action == "": # no op
                    obs = ""
                else:
                    raise NotImplementedError("Unknown action: {}".format(action))
            
            self.agent_scratch_pad += processed_output
            self.agent_scratch_pad += "\n"
            if action == "ask":
                scratch_pad_action = "ask reply"
            else:
                scratch_pad_action = action
            self.agent_scratch_pad += "M: " + scratch_pad_action + ": " + obs + "\n"
            if is_done or i >= self.max_steps:
                if is_done:
                    logger.info("Done")
                else:
                    logger.info("Max steps reached")
                break

            logger.debug("Observation: ", obs, "\n")
            i += 1

    def close(self):
        self.cmd.close()
        
