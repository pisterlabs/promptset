import openai
import pandas as pd
import json
from db import DB

class LLM_API:
    @staticmethod
    def get_completion(prompt, system_content,model="gpt-4"):
        
        messages = [{'role':'system', 'content':system_content},
                    {"role": "user", "content": prompt}]
        
        print(f'Sent to Server messages:\n{messages}')
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
 
        print(f'Response from OpenAI')
        return response.choices[0].message["content"]

    @staticmethod
    def do_label(tabular_data,label):
        
        system_prompt='You are an expert in labelling and speak only JSON.Do not write normal text.'
        prompt=f"""
        You must classify the products delimited by triple backticks:
        1.Provide the output as a JSON string with the following keys only:
        [{{
            "index":,
            "{label}":,
        }},
        {{
            "index":,
            "{label}":,
        }}
        ]
        2.No AI introduction, No AI analysis, Return generated Json data only without backticks, Not human-readable, Remove backticks in output.
        3.If no {label} is found, guess a {label}.
        ```{tabular_data}```"""
        print(prompt)
        response=LLM_API.get_completion(prompt,system_prompt)
        print(f'response:\t{response}')
        response_dict=json.loads(response)
        print(f'response_dict:\t{response_dict}')
        
        #TODO : From_dict to be encapsulated within try/catch block
        return pd.DataFrame.from_dict(response_dict) 

    @staticmethod
    def do_UNSPSC_label(tabular_data,label):
        """
        In-context learning with UNSPSC family labels provided.
        TODO : To merge do_label and do_UNSPSC_label methods together.
        """
        
        system_prompt='You are an expert in labelling and speak only JSON.Do not write normal text.'
        prompt=f"""
        You must classify the products delimited by triple backticks:
        1.Provide the output as a JSON string with the following keys only:
        [
        {{
            "index":,
            "{label}":,
        }},
        {{
            "index":,
            "{label}":,
        }}
        ]
        2.No AI introduction, No AI analysis, Return generated Json data only without backticks, Not human-readable, Remove backticks in output.
        3.If no {label} is found, guess a {label}.
        5.Label the data according to the reference file delimited by triple tilde.
        ```{tabular_data}```
        ~~~
        [
            {{
            "Family(UNSPSC)": "Communications Devices and Accessories"
            }},
            {{
            "Family(UNSPSC)": "Components for information technology or broadcasting or telecommunications"
            }},
            {{
            "Family(UNSPSC)": "Computer Equipment and Accessories"
            }},
            {{
            "Family(UNSPSC)": "Data Voice or Multimedia Network Equipment or Platforms and Accessories"
            }},
            {{
            "Family(UNSPSC)": "Software"
            }},
            {{
            "Family(UNSPSC)": "Printing and publishing equipment"
            }},
            {{
            "Family(UNSPSC)": "Audio and visual presentation and composing equipment"
            }},
            {{
            "Family(UNSPSC)": "Human resources services"
            }},
            {{
            "Family(UNSPSC)": "Legal services"
            }},
            {{
            "Family(UNSPSC)": "Real estate services"
            }},
            {{
            "Family(UNSPSC)": "Business administration services"
            }},
            {{
            "Family(UNSPSC)": "Professional engineering services"
            }},
            {{
            "Family(UNSPSC)": "Computer services"
            }},
            {{
            "Family(UNSPSC)": "Information Technology Service Delivery"
            }},
            {{
            "Family(UNSPSC)": "Advertising"
            }},
            {{
            "Family(UNSPSC)": "Writing and translations"
            }},
            {{
            "Family(UNSPSC)": "Telecommunications media services"
            }},
            {{
            "Family(UNSPSC)": "Information services"
            }}
        ]
        ~~~
        """
        print(prompt)
        response=LLM_API.get_completion(prompt,system_prompt)
        print(f'response:\t{response}')
        response_dict=json.loads(response)
        print(f'response_dict:\t{response_dict}')
        
        #TODO : From_dict to be encapsulated within try/catch block
        return pd.DataFrame.from_dict(response_dict)   
    
    @staticmethod
    def get_sql(nlp_query):
        ''''''
        system_prompt = "you are a text-to-SQL translator. You write sqlite code based on plain-language prompts.Do not have line breaks in the output."
        prompt=f"""
        - Language = sqlite
        - {DB.getColsDB()}
        You are a SQL code translator. Your role is to translate natural language text delimited by triple backticks to sqlite. Your only output should be SQL code. Do not include any other text. Only SQL code.
        Translate ```{nlp_query}``` to a syntactically-correct sqlite query. 
        """
        print(prompt)
        response=LLM_API.get_completion(prompt,system_prompt)
        print(f'response:\t{response}')
        return response



    