from openai import AsyncOpenAI,OpenAI
import openai
import instructor
import json
import uuid
from models.mission_model import ExtractedMission
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

instructor.patch(client)
# ++++++++++++++++++++++++++++++++++ OLD CODE +++++++++++++++++++++++++++++++++++++
# def extract_mission_details(mission):
                            
#     messages = [{
#             "role": "system",
#             "content": """
#                         You are a project manager who has contributed to the creation of hundreds of companies, all in different fields, which allows you to have global expertise. 
#                         """},
#                         {
#             "role": "user",
#             "content": f"As a mission provider on the platform, you benefit from an AI assistant that helps you to extract technical details from a mission so you can frame your needs rapidly and accurately.\
#                 Here is the mission description : {mission}"
#                 }]

#     function = {
#         "name": "mission",
#         "description" : "A function that takes in a mission description and returns a list of technical deductions",
#         "parameters" : {
#             "type" : "object",
#             "properties" : {
#                 "name" : {
#                     "type" : "string",
#                     "description" : "A synthetic name of the mission"
#                 },
#                 "abstract" : {
#                     "type" : "string",
#                     "description" : "A reformulated, synthetic description of the mission"
#                 },
#                 "detail" : {
#                 "type" : "string",
#                 "description" : "An advanced technical reformulation of the mission, highlight important details with the html tags <b> </b and make it readable with <br> </br>. Give maximum details and be as precise as you can on the tech to use and the process" 
#                 },
#                 "roles": {
#                 "type": "array",
#                 "description": "A list of the different required roles to accomplish the mission, roles must be related to tech/developpement/design.",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                     "role_name": {
#                         "type": "string",
#                         "description": "emoji related to the role + name of the role"
#                     },
#                     "skills_required": {
#                         "type": "array",
#                         "items": {
#                         "type": "string"
#                         },
#                         "description": "List of skills required for the role"
#                     },
#                     "reason" : {
#                         "type" : "string",
#                         "description" : "The reason why this role is required, give maximum details and be as precise as you can on the tech to use and the process."
                        
#                     }
#                     },
#                     "required": ["role_name", "skills_required", "reason"]
#                 }
#                 },
#                 "budget": {
#                     "type": "object",
#                     "description": "Budget details of the mission if mentioned in the mission or a fair assessment of the budget",
#                     "properties": {
#                         "total": {
#                             "type": "number",
#                             "description": "The total cost of the mission"
#                         },
#                         "roles_budget": {
#                             "type": "array",
#                             "description": "Budget allocation for each role involved in the mission",
#                             "items": {
#                                 "type": "object",
#                                 "properties": {
#                                     "role_name": {
#                                         "type": "string",
#                                         "description": "Name of the role"
#                                     },
#                                     "allocated_budget": {
#                                         "type": "number",
#                                         "description": "Budget allocated for this role"
#                                     }
#                                 },
#                                 "required": ["role_name", "allocated_budget"]
#                             }
#                         }
#                     },
#                     "required": ["total", "roles_budget"]
#                 }
                
                
                
#                 },
#                 "required": ["name", "abstract", "detail", "roles", "budget"]
            
#             }
#         }

#     response = openai.chat.completions.create(
#         model="gpt-4-1106-preview",
#         messages=messages,
#         functions=[function],
#         function_call={"name": "mission"}, # this forces calling `function`
#     )

#     raw_response = response.model_dump()
#     raw_response['prompt_cost'] = response.usage.prompt_tokens * 0.01/1000
#     raw_response['completion_cost'] = response.usage.completion_tokens * 0.03/1000
#     raw_response['cost'] = {'prompt' : raw_response['prompt_cost'], 'completion' : raw_response['completion_cost'], 'total' : raw_response['prompt_cost'] + raw_response['completion_cost']}
#     raw_response['id'] = uuid.uuid1()

#     mission_dict = json.loads(response.model_dump()['choices'][0]['message']['function_call']['arguments'])
#     mission_dict['id'] = uuid.uuid1()
#     mission_dict['created'] = raw_response['created']
#     mission_dict['metadata_id'] = raw_response['id']

#     return mission_dict, raw_response

# ++++++++++++++++++++++++++++++++++ OLD CODE +++++++++++++++++++++++++++++++++++++


def extract_mission_details(mission: str, model = "gpt-4-1106-preview") -> ExtractedMission:

    if mission == "test":
        model="gpt-3.5-turbo"

    response = client.chat.completions.create(
        model=model,
        response_model=ExtractedMission,
        messages = [{
                "role": "system",
                "content": """
                            You are a project manager who has contributed to the creation of hundreds of companies, all in different fields, which allows you to have global expertise. 
                            """},
                            {
                "role": "user",
                "content": f"As a mission provider on the platform, you benefit from an AI assistant that helps you to extract technical details from a mission so you can frame your needs rapidly and accurately.\
                    Here is the mission description : {mission}"
                    }]
                    )
    
    
    raw_response = response._raw_response.model_dump()
    raw_response['prompt_cost'] = response._raw_response.usage.prompt_tokens * 0.01/1000
    raw_response['completion_cost'] = response._raw_response.usage.completion_tokens * 0.03/1000
    raw_response['cost'] = {'prompt' : raw_response['prompt_cost'], 'completion' : raw_response['completion_cost'], 'total' : raw_response['prompt_cost'] + raw_response['completion_cost']}
    raw_response['id'] = uuid.uuid1()

    mission_dict = json.loads(response._raw_response.model_dump()['choices'][0]['message']['function_call']['arguments'])
    mission_dict['id'] = uuid.uuid1()
    mission_dict['created'] = raw_response['created']
    mission_dict['metadata_id'] = raw_response['id']

    return mission_dict, raw_response