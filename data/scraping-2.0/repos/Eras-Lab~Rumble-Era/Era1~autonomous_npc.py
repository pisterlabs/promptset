import openai
import os
import json
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key


class AutonomousNPC:
    def __init__(self, model_name, temperature, api_key, templates):
        openai.api_key = api_key
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.templates = templates  # This will be a list of dictionaries
        self.expected_keys = ["todo1", "todo2"]

    def is_json(self, myjson):
        try:
            json_object = json.loads(myjson)
        except ValueError as e:
            return False
        return True

    def validate_format(self, json_str):
        #Define the JSON format

        if self.is_json(json_str):
            json_object = json.loads(json_str)

            if all(key in json_object for key in self.expected_keys):
                return True
            else:
                return False
        else:
            return False

    def run(self, input_params):
        responses = []
        counter = 0
        max_attempts = 1

        for template in self.templates:
            # Set up the prompts and llmchain
            system_message_prompt = SystemMessagePromptTemplate.from_template(template['system_template'])
            human_message_prompt = HumanMessagePromptTemplate.from_template(template['human_template'])
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
            llm_chain = LLMChain(llm=self.llm, prompt=chat_prompt)
            input_params["response"] = responses
            # Call the LLM
            response = llm_chain.run(**input_params)
            responses.append(response)

            # Check validatition
            response_valid = False
            attempt = 0
            response = ''
            while not response_valid and attempt < max_attempts:
                # update some input param values
                input_params["response"] = responses
                # Call the LLM
                response = llm_chain.run(**input_params)
                print("Response: ", response)
                response_valid = self.validate_format(response)
                attempt += 1
                print("Attempt: ", attempt)

            responses.append(response)

        
        valid_responses_ratio = sum([self.validate_format(response) for response in responses]) / len(responses)
        #print(f"Valid Responses Ratio: {valid_responses_ratio}")
        #print(f"Counter: {counter / len(responses)}")
        return responses

#1 Just decides the todo list:
#2 Analysize the todo list and decide on the next action
#3 Analysize response from last action

# What kind of informations should the AI have when deciding on decisions?
# What kind memory or data.

# templates = [
#     { 
#         "system_template": "Given this problem: {problem} and observations: {observations} , given that you want to become stronger, make a todo list. The answer should be in JSON format only, here is an example: '{{\"todo1\": \"do pushups\", \"todo2\": \"do 100 sit ups\", \"todo3\": \"do 100 squats\", \"todo4\": \"run 10km\"}}'",
#         "human_template": "{text}"
#     },
#     {
#         "system_template": "Please analyze this list which is in a JSON format: {response} and respond if you think this is the best way and change the list if necessary. The answer should be in JSON format only and nothing more, here is an example: '{{\"opinion\": \"I believe the json file list is good\", \"todo1\": \"do pushups\", \"todo2\": \"do 100 sit ups\", \"todo3\": \"do 100 squats\", \"todo4\": \"run 10km\"}}'",
#         "human_template": ""
#     },
#     {
#         "system_template": "Please analyze this list which is in a JSON format: {response} and respond if you think this is the best way and change the list if necessary. The answer should be in JSON format only and nothing more, here is an example: '{{\"opinion\": \"I believe the json file list is good\", \"todo1\": \"do pushups\", \"todo2\": \"do 100 sit ups\", \"todo3\": \"do 100 squats\", \"todo4\": \"run 10km\"}}'",
#         "human_template": ""
#     },
#     {
#         "system_template": "Please analyze this list which is in a JSON format: {response} and respond if you think this is the best way and change the list if necessary. The answer should be in JSON format only and nothing more, here is an example: '{{\"opinion\": \"Yes, I think the best way to represent the workout routine instructions is to remove the outer list brackets and convert the inner object into a JSON string. Here is the updated JSON:\",\"todo1\": \"do pushups\", \"todo2\": \"do 100 sit ups\", \"todo3\": \"do 100 squats\", \"todo4\": \"run 10km\"}}'",
#         "human_template": ""
#     },
# ]

# autonomous_npc = AutonomousNPC(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key, templates=templates)

# input_params = {"problem": "I need an exercise plan", "observations": "I am 80kgs", "response": "", "text": ""}
# autonomous_npc.run(input_params)