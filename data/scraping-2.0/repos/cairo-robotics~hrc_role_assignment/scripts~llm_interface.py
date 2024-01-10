import openai
import yaml
import json

from oai_agents.common.subtasks import Subtasks

FORMATTING_PROMPT = "Please format your last response as a Python dictionary. Use the format {'Division name' : {'Role name' : ['Task list']}}"

# SAMPLE_GPT_OUTPUT = {
#   "id": "chatcmpl-7gJb4ll8OC1G9ifdPzbjkBx95l99A",
#   "object": "chat.completion",
#   "created": 1690319462,
#   "model": "gpt-3.5-turbo-0613",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "{\n  'Division 1': {\n    'Chef': ['Grabbing an onion from dispenser', 'Putting onion in pot', 'Grabbing dish from dispenser', 'Placing dish closer to pot', 'Serving the soup'],\n    'Sous Chef': ['Grabbing a tomato from dispenser', 'Putting tomato in pot', 'Grabbing dish from counter', 'Getting the soup', 'Grabbing soup from counter', 'Placing soup closer']\n  },\n  'Division 2': {\n    'Prep Cook': ['Grabbing an onion from dispenser', 'Grabbing a tomato from dispenser', 'Putting onion in pot', 'Putting tomato in pot'],\n    'Server': ['Grabbing dish from dispenser', 'Grabbing dish from counter', 'Placing dish closer to pot', 'Getting the soup', 'Grabbing soup from counter', 'Placing soup closer', 'Serving the soup']\n  },\n  'Division 3': {\n    'Cook': ['Grabbing an onion from dispenser', 'Grabbing a tomato from dispenser', 'Putting onion in pot', 'Putting tomato in pot', 'Getting the soup'],\n    'Waiter': ['Grabbing dish from dispenser', 'Grabbing dish from counter', 'Placing dish closer to pot', 'Grabbing soup from counter', 'Placing soup closer', 'Serving the soup']\n  },\n  'Division 4': {\n    'Food Prep': ['Grabbing an onion from dispenser', 'Grabbing a tomato from dispenser', 'Putting onion in pot', 'Putting tomato in pot', 'Getting the soup'],\n    'Service': ['Grabbing dish from dispenser', 'Grabbing dish from counter', 'Placing dish closer to pot', 'Grabbing soup from counter', 'Placing soup closer', 'Serving the soup']\n  }\n}"
#       },
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 514,
#     "completion_tokens": 372,
#     "total_tokens": 886
#   }
# }

SAMPLE_GPT_OUTPUT = {'Option 1': {'Player 1 (Preparation Role)': ['Grabbing an onion from dispenser', 'Grabbing an onion from counter', 'Putting onion in pot', 'Placing onion closer to pot', 'Grabbing a tomato from dispenser', 'Grabbing a tomato from counter', 'Putting tomato in pot', 'Placing tomato closer to pot', 'Grabbing a cabbage from dispenser', 'Grabbing a cabbage from counter', 'Putting cabbage in pot', 'Placing cabbage closer to pot'], 'Player 2 (Cooking Role)': ['Grabbing a fish from dispenser', 'Grabbing a fish from counter', 'Putting fish in pot', 'Placing fish closer to pot', 'Grabbing dish from dispenser', 'Grabbing dish from counter', 'Placing dish closer to pot', 'Getting the soup', 'Grabbing soup from counter', 'Placing soup closer', 'Serving the soup']}, 'Option 2': {'Player 1 (Preparation Role)': ['Grabbing an onion from dispenser', 'Grabbing an onion from counter', 'Putting onion in pot', 'Placing onion closer to pot', 'Grabbing a tomato from dispenser', 'Grabbing a tomato from counter', 'Putting tomato in pot', 'Placing tomato closer to pot'], 'Player 2 (Cooking Role)': ['Grabbing a cabbage from dispenser', 'Grabbing a cabbage from counter', 'Putting cabbage in pot', 'Placing cabbage closer to pot', 'Grabbing a fish from dispenser', 'Grabbing a fish from counter', 'Putting fish in pot', 'Placing fish closer to pot', 'Grabbing dish from dispenser', 'Grabbing dish from counter', 'Placing dish closer to pot', 'Getting the soup', 'Grabbing soup from counter', 'Placing soup closer', 'Serving the soup']}, 'Option 3': {'Player 1 (Preparation Role)': ['Grabbing an onion from dispenser', 'Grabbing an onion from counter', 'Putting onion in pot', 'Placing onion closer to pot', 'Grabbing a cabbage from dispenser', 'Grabbing a cabbage from counter', 'Putting cabbage in pot', 'Placing cabbage closer to pot'], 'Player 2 (Cooking Role)': ['Grabbing a tomato from dispenser', 'Grabbing a tomato from counter', 'Putting tomato in pot', 'Placing tomato closer to pot', 'Grabbing a fish from dispenser', 'Grabbing a fish from counter', 'Putting fish in pot', 'Placing fish closer to pot', 'Grabbing dish from dispenser', 'Grabbing dish from counter', 'Placing dish closer to pot', 'Getting the soup', 'Grabbing soup from counter', 'Placing soup closer', 'Serving the soup']}}

class GPTRolePrompter:
    def __init__(self):
        with open("config.yaml", "r") as stream:
            self.api_key = yaml.safe_load(stream)["OPENAI_API_KEY"]
            openai.api_key = self.api_key

    def get_llm_response(self, prompt, temperature=0.0):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=temperature
        )
        return response

    def generate_role_prompt(self, n_players, subtask_list):
        prompt = '''{}
                    I want to divide these tasks between {} players. \
                    Suggest a few ways we could divide these tasks into roles and what those roles would be called.\
                    '''.format(subtask_list, str(n_players))
        messages = [
            {"role": "system", "content": "You are a task manager for kitchen staff."},
            {"role": "user", "content": prompt}
        ]
        return messages
    
    def query_for_role_divisions(self, subtask_list):
        query = self.generate_role_prompt(2, subtask_list)
        response = self.get_llm_response(query)
        text_response = response["choices"][0]["message"]["content"]

        # update query and request for format as python dictionary
        query.append({"role": "assistant", "content": text_response})
        query.append({"role": "user", "content": FORMATTING_PROMPT})
        formatted_response = self.get_llm_response(query)
        formatted_response = self.process_gpt_response(formatted_response)

        return formatted_response
    
    def process_gpt_response(self, response):
        try:
            res = json.loads(response["choices"][0]["message"]["content"].replace("'", "\""))
        except json.decoder.JSONDecodeError:
            print("Error: response from model API was not formatted as a Python dictionary.")
            print("Response: {}".format(response["choices"][0]["message"]["content"]))
            return None
        return res

if __name__ == "__main__":
    gpt = GPTRolePrompter()
    print(gpt.query_for_role_divisions(Subtasks.HUMAN_READABLE_ST))
    # print(json.loads(SAMPLE_GPT_OUTPUT["choices"][0]["message"]["content"].replace("'", "\"")))