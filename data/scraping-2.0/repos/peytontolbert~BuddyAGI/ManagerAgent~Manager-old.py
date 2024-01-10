import requests
import openai
import time
from typing import Dict, Any, Optional, Union, List
import os
import json
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Manager agent is designed to handle communication between the user and autonomous agents aswell as communication between agents themselves
class ManagerAgent:

    def __init__(self):
        self.messages = []
        self.activetasks = []
        self.completedtasks = []
        pass

    def run(self):
        self.checkmessages()
        print(self.messages)
        while self.messages:
            print(f"self.messages: {self.messages}")
            message = self.messages.pop(0)  # Remove and return the first message
            sender = message['sender']
            reply = message['message']
            task_id = message['task_id']
            task = self.activetasks.get(task_id)
            receiver = message['receiver']
            agents = message['agents']
            if task is None:
                print("generating a new task")
                newtask = self.generatenewtask(reply)
                self.sendnewtask(newtask, task_id, sender, agents)
            else:
                if sender == 'Buddy':
                    print("handling user reply to a task question")
                    self.handlereply(reply, task_id, sender, receiver)
                else:
                    result = self.handleagentreply(reply, task_id, sender)
                    parsed_message = json.loads(result)
                    next_recipient = parsed_message["next_recipient"]
                    questions = parsed_message["questions"]
                    task = parsed_message["task"]
                    task_completion = parsed_message["complete"]
                    if task_completion == True:
                        print("Task complete")
                        self.completedtasks.append(task_id)
                        self.sendreply(result, next_recipient, task_id, complete=True)
                    self.handlereply(result, task_id, sender, receiver)

        time.sleep(10)

    def generatenewtask(self, chat):
        parsed_message = json.loads(chat)
        agents = parsed_message["agents"]
        task = parsed_message["task"]
        result = self.generatetaskprompt(agents, task)
        return result


    def checkmessages(self):
        try:
            response = requests.get("http://localhost:5000/managermessages")
            if response.status_code == 200:
                self.messages.append(response.text)
            else:
                print("no new messages")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
    

    def save_agent(self) -> None:
        episodic_memory_dir = f"{self.dir}/episodic_memory"
        filename = f"{self.dir}/agent_data.json"
        self.episodic_memory.save_local(path=episodic_memory_dir)

        data = {"name": self.agent_name,
                "episodic_memory": episodic_memory_dir
                }
        with open(filename, "w") as f:
            json.dump(data, f)


    def load_agent(self) -> None:
        absolute_path = self._get_absolute_path()
        if not "agent_data.json" in os.listdir(absolute_path):
            self.ui.notify("ERROR", "Agent data does not exist.", title_color="red")

        with open(os.path.join(absolute_path, "agent_data.json")) as f:
            agent_data = json.load(f)
            try:
                self.episodic_memory.load_local(agent_data["episodic_memory"])
            except Exception as e:
                self.ui.notify(
                    "ERROR", "Episodic memory data is corrupted.", title_color="RED")
                raise e
            else:
                self.ui.notify(
                    "INFO", "Episodic memory data is loaded.", title_color="GREEN")

    def generatetaskprompt(self, agents, task):
        systemprompt = """You are an AI manager agent.
        You are given a list of potential agents to communicate with.
        The first step is to initiate the conversation with all of the agents on the task at hand. The agents will give their input on the task which will be handled in a multi-turn conversation.
        """
        taskprompt = """
        [TASK]
        {task}
        [AGENTS].
        {agents}
        Create a response to initiate the conversation and ask the agents for their input.
        [RESPONSE]
        """
        prompt = taskprompt.format(agents=agents, task=task)
        prompt_messages = [{"role": "system", "content": systemprompt}, {"role": "user", "content": prompt}]
        results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=prompt_messages)
        result = results['choices'][0]['message']['content']
        print(result)
        return result
    
    def handleagentreply(self, reply, task_id, sender):
        lastmessage = reply[-1]
        systemprompt = """You are an autonomous AI manager. Your job is the intent of the agent's message.
        They will be either asking clarifying questions to Buddy, sending a message for communication with another agent, or updating task completion with Buddy.
        If the task is complete, you may verify send a task completion option aswell
        Reply in JSON format following the example:
        [EXAMPLE]
        {
            "next_recipient": "Buddy/agent_name",
            "questions:" ["Optional question1", "Optional question2"],
            "task": "any specific task to complete"
            "complete": "True/False"
        }"""

        messages = [{"role": "system", "content": systemprompt}, {"role": "user", "content": lastmessage}]
        results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages)
        result = results['choices'][0]['message']['content']
        print(result)
        return result

    def sendreply(self, reply, recipient, identifier, complete=False):
        url = "http://localhost:5000/aitask"
        data = {"message": reply, "recipient": recipient, "task_id": identifier} if complete==False else {"message": reply, "user": recipient, "task_id": identifier, "complete_status": True}
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                print("message sent")
            else:
                print("message not sent")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")



    def sendnewtask(self, newtaskprompt, task_id, agents):
        url = "http://localhost:5000/newtaskintro"
        data = {"message": newtaskprompt, "task_id": task_id, "agents": agents}
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                print("message sent")
            else:
                print("message not sent")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
