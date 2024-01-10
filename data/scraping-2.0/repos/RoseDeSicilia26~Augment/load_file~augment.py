import pandas as pd
import openai
import os
import re
import ipywidgets as widgets

openai.api_key = 'api_key'

class Augment():

    def __init__(self):
        self.schema : dict
        self.treatment : str
        self.outcome : str
        self.confounders : dict

    # expeting txt
    def load_text_file(self, file_path, delimiter = "|"):

        self.schema = {}

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace

                if delimiter in line:
                    var_name, description = line.split(delimiter, 1)
                    var_name = var_name.strip()
                    description = description.strip()
                else:
                    var_name = line.strip()
                    description = ""

                self.schema[var_name] = description


    #expecting pandas dataframe
    def load_data(self, df: pd.DataFrame):

        column_names = df.columns.to_list()

        # Create a dictionary with column names as keys and empty string
        self.schema = {col: "" for col in column_names}

    # returns schema keys and values as a string
    def get_schema(self):
        # Create a list of strings in the format 'key: value'
        items = [f'{key}: {value}' for key, value in self.schema.items()]

        # Join the list of strings into one string with '\n' as the separator
        result = '\n'.join(items)

        return result
    
    # get schema keys as a string
    def get_schema_keys(self):

        # save all the keys in a list 
        keys = list(self.schema.keys())

        # convert the list of keys to a string, with each key separated by a comma and a space
        keys_str = ', '.join(keys)

        return keys_str
    
     # returns schema keys and values as a string
    def get_confounders(self):
        # Create a list of strings in the format 'key: value'
        items = [f'{key}: {value}' for key, value in self.confounders.items()]

        # Join the list of strings into one string with ', ' as the separator
        result = '\n'.join(items)

        return result

    # generate and validate new descriptions 
    def augment_descriptions(self, auto : bool = True):

        # iterate through each var and it's associated description 
        for var, desc in self.schema.items():
            
            # get back the system and user prompts based on this key value pair
            system_prompt, user_prompt = self.data_prompt(var, desc, auto)

            # get back a list of descriptions
            descriptions = self.generate_descriptions(system_prompt, user_prompt)

            # append description that came with schema - if any came at all
            if(desc != ""):
                descriptions.append(desc)

            if bool == False:
                # print descriptions
                print(f"Description options for {var}.")
                for i in range(len(descriptions)):
                    print(f"{(i+1)}) {descriptions[i]}")
                
                # add try catch
                # add comment option
                # ask user to select description
                selected_descr = ""
                while selected_descr == "":                
                    selected_descr = input("Select the description that best matches the column by typing in the number (i.e. 1, 2, 3, ...)") #\nIf none match, then explain where they fail to regenerate improved descriptions.")
                    if(selected_descr == "" ):
                        print("Please select a description")
                print("\n")
                # save selected description to the dictionary
                self.schema[var] = descriptions[int(selected_descr)]
            
            else:
                #print(f"{var}\n{descriptions[0]}\n")

                self.schema[var] = descriptions[0]



    # generate and validate gpt identified confounders
    def augment_relationships(self, treatment: str, outcome: str, auto: bool = True):

        self.treatment = treatment
        self.outcome = outcome
        
        # get back the system and user prompts based on this key value pair
        system_prompt, user_prompt = self.relationships_prompt()

        # get back a list of descriptions
        relationships = self.generate_relationships(system_prompt, user_prompt)

        # get user validation
        #print(f"These are the variables confounding the relationship between the treatment {self.treatment} and outcome {self.outcome}.\n")
        # iterate over generated confounders and ask user to validate or discard  
        if auto == False:    
            for confounder, reason in relationships.items():
                
                validated : int = None
                #print(f"Confounder: {confounder}\nReason: {reason}\n")

                #add try catch
                validated = int(input("Does this confounder make sense? Input 1 or 0"))
                while validated != 0 or validated != 1:
                    validated = int(input("Please input a 1 or a 0."))
                
                if validated == 1:
                    self.confounders[confounder] = reason

        else:
            self.confounders = relationships


    def data_prompt(self, var: str, desc: str, auto: bool):

        user_prompt = ""

        if auto:
            system_prompt = "You are a helpful causality assistant with expertise in causal inference. I will provide you with the column names for a given dataset in comma separated order. I will then ask you to write me a description for a selected column. Let's take it step by step to make sure the description is relevant, succinct, and clear. Wrap the description in the form <description></description>."
    
        else:
            system_prompt = "You are a helpful causality assistant with expertise in causal inference. I will provide you with the column names for a given dataset in comma separated order. I will then ask you to write me possible descriptions for a selected column. Let's take it step by step to make sure the description is relevant, succinct, and clear. Wrap the description in the form <description></description>."

        if desc != "":
            user_prompt =  f"Here is the schema, the selected column, and an example description for that column. Schema\n{self.get_schema_keys()}\nSelected column\n{var}\nExample description\n{desc}"

        else:
            user_prompt = f"Here is the schema and the selected column. Schema\n{self.get_schema_keys()}\nSelected column\n{var}"
            
        return user_prompt, system_prompt
    
    def relationships_prompt(self):
        system_prompt = f"You are a helpful causality assistant with expertise in causal inference. I will provide you with the dataset schema (where each variable description) that I am using to study the causal relationship between the treatment {self.treatment} and outcome {self.outcome}. I will then ask you to identify the variables confounding that relationship, where a confounding variable is one that is a direct cause/parent of both the treatment {self.treatment} and the outcome {self.outcome}. I will also ask that you explain your reasoning. Wrap each confounder and reasoning in the form <confounder></confounder><reason></reason>"

        user_prompt = f"Here is the dataset schema that I am using to study the causal relationship between treatment {self.treatment} ({self.schema[self.treatment]}) and outcome {self.outcome} ({self.schema[self.outcome]}).\nHere is the schema\n{self.get_schema()}"

        return user_prompt, system_prompt 


    # improve input/output with 'guidance'
    def generate_descriptions(self, system_prompt: str, user_prompt: str):

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            n = 1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        descriptions = re.findall(r"<description>(.*?)</description>", completion.choices[0].message.content)

        # return extracted list of strings
        return descriptions
    

    def generate_relationships(self, system_prompt: str, user_prompt: str):
        
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            n = 1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        #print(completion.choices[0].message.content)
        # Find all occurrences of confounders and reasonings
        confounders = re.findall(r'<confounder>(.*?)</confounder>', completion.choices[0].message.content)
        reasonings = re.findall(r'<reason>(.*?)</reason>', completion.choices[0].message.content)

        # Combine confounders and reasonings into a dictionary
        relationships = dict(zip(confounders, reasonings))

        # Create a list of strings in the format 'key: value'
        items = [f'{key}: {value}' for key, value in relationships.items()]

        # Join the list of strings into one string with ', ' as the separator
        result = '\n'.join(items)

        #print(result)

        # return extracted list of relationships
        return relationships



