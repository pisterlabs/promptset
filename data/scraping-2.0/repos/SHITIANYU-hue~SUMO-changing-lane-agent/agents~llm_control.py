import openai
import os
import time
import pandas as pd
import json
os.environ["OPENAI_API_KEY"] = "sk-"
openai.api_key = os.environ["OPENAI_API_KEY"]
import openai
import time

class llmagent:
    def __init__(self):
        self.api_key = 'sk-'  # Replace with your OpenAI API key
    
    def generate_chat_completion(self, prompt):
        try:
            openai.api_key = self.api_key
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': prompt}]
            )
            usage = response["usage"]["total_tokens"]
            return response.choices[0].message.content, usage

        except openai.error.RateLimitError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self.generate_chat_completion(prompt)

        except openai.error.ServiceUnavailableError as e:
            retry_time = 10  # Adjust the retry time as needed
            print(f"Service is unavailable. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self.generate_chat_completion(prompt)

        except openai.error.APIError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
            print(f"API error occurred. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self.generate_chat_completion(prompt)

        except OSError as e:
            retry_time = 5  # Adjust the retry time as needed
            print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self.generate_chat_completion(prompt)
        
    def generate_decision(self, state):
        description = '''
        {
            'Ego Vehicle': {
                'Speed': ego vehicle's current speed,
                'Target Speed': target speed set for the ego vehicle
            },
            'Leading Vehicle in Same Lane': {
                'Speed': speed of the leading vehicle in the same lane,
                'Headway': distance between the ego vehicle and the leading vehicle in the same lane
            },
            'Leading Vehicle in Right Lane': {
                'Speed': speed of the leading vehicle in the right lane,
                'Headway to Ego Vehicle': self.calculate_headway_to_ego(lead_right)
            },
            'Leading Vehicle in Left Lane': {
                'Speed': speed of the leading vehicle in the left lane,
                'Headway to Ego Vehicle': self.calculate_headway_to_ego(lead_left)
            },
            'Following Vehicle in Right Lane': {
                'Speed': speed of the following vehicle in the right lane,
                'Headway to Ego Vehicle': self.calculate_headway_to_ego(follow_right)
            },
            'Following Vehicle in Left Lane': {
                'Speed': speed of the following vehicle in the left lane,
                'Headway to Ego Vehicle': self.calculate_headway_to_ego(follow_left)
            }
            'safe_constraint check:{
            'change right': whether or not able to change right,
            'change left': whether or not able to change left
            }
        }
        '''
        input_=f"""
        You are a driving assitant. You are designed to assist the vehicle `ego` in making driving decisions based on scenario information provided by humans. \
        You understand the state of the vheicle `ego` and suggests possible actions for the vehicle based on the state. Further, you can evaluate the actions proposed in the previous step and thus select the most appropriate action. \
        In the following, you will be given some observations from the sensors on the vehicle. \
        You only need to make inferences based on the available information. You do not have to assume or consider dangers that have NOT occurred.\
        You need to analyze the scenario step by step \
        You should NOT assume scenarios that are not happening, but only for the current observation.\
        Here is the observation from the sensors: ```{state}```\
        The meaning of the observation is: {description}\
        your decision is two dimensional, longitudinal is: [accelerate(1), remain(0), decelerate(-1)], lateral is [change_left(2), stay_in_same_lane(0), change_right(1)]
        generate your decision, e.g [0,1]
        """,

        output,_=self.generate_chat_completion(str(input_))
        print('llm output',output)
        lane, acceleration=0,0
        return lane, acceleration
