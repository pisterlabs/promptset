# import json
# import os
# import uuid

# from dotenv import load_dotenv

# load_dotenv()

# import autogen
# from .openai import OpenAIAgent


# class AutoGenAgent:
#     config_list = [
#         {
#             "model": "gpt-4",
#             "api_key": os.getenv("OPENAI_API_KEY"),
#         },
#         {
#             "model": "gpt-4-32k",
#             "api_key": os.getenv("OPENAI_API_KEY"),
#         },
#         {
#             "model": "gpt-3.5-turbo",
#             "api_key": os.environ.get("OPENAI_API_KEY"),
#         },
#     ]

#     def __init__(self, config_path=None):
#         aiteam_config = {}
#         if config_path:
#             with open(config_path, "r") as f:
#                 aiteam_config = json.load(f)
#         self.seed = aiteam_config.get("seed", 42)
#         self.max_round = aiteam_config.get("max_round", 20)
#         self.temperature = aiteam_config.get("temperature", 0)
#         self.input_mode = aiteam_config.get("input_mode", "NEVER")
#         self.max_reply = aiteam_config.get("max_reply", 10)
#         self.agents = {}
#         self.proxy = autogen.UserProxyAgent(
#             name="user_proxy",
#             human_input_mode=self.input_mode,
#             max_consecutive_auto_reply=self.max_reply,
#             is_termination_msg=lambda x: x.get("content", "")
#             .rstrip()
#             .endswith("TERMINATE"),
#             code_execution_config={
#                 "work_dir": "tmp",
#                 "use_docker": False,  # set to True or image name like "python:3" to use docker
#             },
#         )

#         agents = aiteam_config.get(
#             "agents",
#             [
#                 {
#                     "name": "Assistant",
#                     "role": "You are a helpful, encouraging, and genial AI Assistant ready to help with any task.",
#                 }
#             ],
#         )
#         for agent in agents:
#             self.create_assistant(agent["name"], agent["role"])

#     @property
#     def solution(self):
#         self.proxy.send(
#             "summarize the solution in an easy-to-understand way", self.manager
#         )
#         # return the last message the proxy received
#         last_message = self.proxy.last_message()
#         return last_message["content"] if last_message.get("content") else None

#     def create_agent(self, name, system_message):
#         assistant = autogen.AssistantAgent(
#             name=name,
#             system_message=system_message,
#             llm_config={
#                 "seed": self.seed,
#                 "config_list": self.config_list,
#                 "temperature": self.temperature,
#             },
#         )
#         self.agents[name] = assistant
#         return self

#     def generate(self, message):
#         # the assistant receives a message from the user_proxy, which contains the task description

#         groupchat = autogen.GroupChat(
#             agents=list(self.agents.values()),
#             messages=[message],
#             max_round=self.max_round,
#         )
#         self.manager = autogen.GroupChatManager(
#             groupchat=groupchat,
#             llm_config={
#                 "seed": self.seed,
#                 "config_list": self.config_list,
#                 "temperature": self.temperature,
#             },
#         )
#         self.proxy.initiate_chat(self.manager, message=message)

#     def generate_image(self, prompt, **kwargs):
#         pass

#     def generate_json(
#         self,
#         text,
#         functions,
#         primer_text="",
#     ):
#         pass

#     def generate_text(self, text, primer_text=""):
#         pass

#     def summarize_text(self, text, primer=""):
#         pass
