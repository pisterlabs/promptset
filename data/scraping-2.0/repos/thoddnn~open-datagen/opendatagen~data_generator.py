
from dotenv import load_dotenv
import numpy as np
import time
import random
import re
import json
import requests
from urllib.parse import quote
from re import findall
from typing import Dict, List, Union
from opendatagen.utils import dict_to_string, load_file, write_to_csv, generate_context_from_json, extract_website_details, create_type_message, find_strings_in_brackets
from opendatagen.utils import snake_case_to_title_case, title_case_to_snake_case
from opendatagen.utils import extract_content_from_internet, clean_string
from opendatagen.anonymizer import Anonymizer
from opendatagen.model import OpenAIChatModel, OpenAIInstructModel, OpenAIEmbeddingModel, ModelName, MistralChatModel, LlamaCPPModel
from opendatagen.template import Template, Variable, Variations, create_variable_from_name
from opendatagen.utils import function_to_call
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import uuid

load_dotenv()

class DataGenerator:

    output_array = []

    def __init__(self, template:Template):

        self.template = template

    def extract_variable_from_string(self, text:str):
        return findall(r'\{(.*?)\}', text)

    def extract_variable_dict_from_string(self, text:str):
        
        list_of_variables = findall(r'\{(.*?)\}', text)

        result = {}

        for variable_id, variable in self.template.variables.items():
            
            if variable_id in list_of_variables:
                result[variable_id] = variable

        return result
    


    def anonymize_text(self, text_to_anonymize):

        # Example usage:
        anonymizer = Anonymizer()

        anonymized_text = anonymizer.anonymize(text_to_anonymize)

        return anonymized_text
    
    def contextual_generation(self, prompt_text:str, variables:list, current_variation_dict:dict, fixed_variables: Dict[str, Variable], completion:str=None, parent_id:str=None):

        # This will be the list to collect all dictionaries
        result = []

        if not variables:
            # No more variables to process, generate final variation
            return [current_variation_dict.copy()]
        
        # Get the next variable
        next_var = variables[0]
        remaining_variables = variables[1:]

        if completion:

            formatted_template = completion.format(**{var: current_variation_dict.get(var, f'{{{var}}}').value if hasattr(current_variation_dict.get(var, f'{{{var}}}'), 'value') else current_variation_dict.get(var, f'{{{var}}}') for var in re.findall(r'\{(.*?)\}', completion)})
            current_completion = formatted_template.split(f'{{{next_var}}}')[0] + f'{{{next_var}}}'  
            current_prompt = prompt_text

        else:

            formatted_template = prompt_text.format(**{var: current_variation_dict.get(var, f'{{{var}}}').value if hasattr(current_variation_dict.get(var, f'{{{var}}}'), 'value') else current_variation_dict.get(var, f'{{{var}}}') for var in re.findall(r'\{(.*?)\}', prompt_text)})
            current_prompt = formatted_template.split(f'{{{next_var}}}')[0] + f'{{{next_var}}}' 
            current_completion = None 

        variable = fixed_variables[next_var]

        variations = self.generate_variable(prompt_text=current_prompt,
                                                       completion_text=current_completion,
                                                       current_variable=variable,
                                                       variable_id_string=next_var,
                                                       parent_id=parent_id)

        for id, variation in variations.items():
            # Update the current variations dictionary with the new variation
            updated_variation_dict = current_variation_dict.copy()

            updated_variation_dict[next_var] = variation

            # Recursively process the remaining variables
            # and extend the all_variation_dicts list with the results
            result.extend(self.contextual_generation(
                prompt_text=prompt_text,
                completion=completion,
                variables=remaining_variables,
                current_variation_dict=updated_variation_dict,
                fixed_variables=fixed_variables,
                parent_id=id
            ))

        # Return the list of all variation dictionaries generated
        return result

    def generate_variable(self, prompt_text:str, current_variable:Variable, variable_id_string:str, completion_text:str=None, parent_id:str=None):

        generation_number = current_variable.generation_number

        variations = {}

        if current_variable.get_value_from_localfile:

            for _ in range(generation_number):

                generated_value = current_variable.get_value_from_localfile.get_content_from_file()

                if parent_id:

                    new_id = str(uuid.uuid4())

                    new_value = Variations(id=new_id, parent_id=parent_id, value=generated_value)

                    current_variable.values[new_id] = new_value

                    self.template.variables[new_id]

                    variations[new_id] = new_value

                    self.template.variables[variable_id_string].values[new_id] = new_value

                else:

                    id_loop = str(uuid.uuid4())

                    new_value = Variations(id=id_loop, parent_id=id_loop, value=generated_value)

                    current_variable.values[id_loop] = new_value

                    variations[id_loop] = new_value

                    self.template.variables[variable_id_string].values[id_loop] = new_value

            return variations
        

        if current_variable.get_value_from_huggingface:

            for _ in range(generation_number):

                generated_value = current_variable.get_value_from_huggingface.get_random_value_from_dataset()

                if parent_id:

                    new_id = str(uuid.uuid4())

                    new_value = Variations(id=new_id, parent_id=parent_id, value=generated_value)

                    current_variable.values[new_id] = new_value

                    self.template.variables[new_id]

                    variations[new_id] = new_value

                    self.template.variables[variable_id_string].values[new_id] = new_value

                else:

                    id_loop = str(uuid.uuid4())

                    new_value = Variations(id=id_loop, parent_id=id_loop, value=generated_value)

                    current_variable.values[id_loop] = new_value

                    variations[id_loop] = new_value

                    self.template.variables[variable_id_string].values[id_loop] = new_value

            return variations

        if completion_text:
            initial_variation_prompt = load_file(path="files/completion.txt")
        else:
            initial_variation_prompt = load_file(path="files/generation.txt")

        temp_variation_prompt = initial_variation_prompt

        name = current_variable.name

        if current_variable.note:
            note = random.choice(current_variable.note)
        else:
            note = ""
        
        rag_content = ""

        if current_variable.source_localfile:
            current_variable.load_local_file()
        elif current_variable.source_localdirectory:
            current_variable.load_local_directory()
        elif current_variable.source_internet:
            current_variable.load_internet_source()
        elif current_variable.source_huggingface:
            current_variable.load_huggingface_dataset()

        if current_variable.rag_content:
            rag_content = f"Here are some examples that might help you:\n\n{current_variable.rag_content}"

        last_values_list = []
        last_values = ""

        for _ in range(generation_number):

            current_model = random.choice(current_variable.models).get_model()

            if isinstance(current_model, OpenAIInstructModel) or isinstance(current_model, LlamaCPPModel): 
                if current_model.start_with:
                    start_with = random.choice(current_model.start_with)
                else:
                    start_with = ""
            else:
                start_with = ""

            if current_variable.source_localfile:
                current_variable.load_local_file()
            elif current_variable.source_localdirectory:
                current_variable.load_local_directory()
            elif current_variable.source_internet:
                current_variable.load_internet_source()
            elif current_variable.source_huggingface:
                current_variable.load_huggingface_dataset()

            if current_variable.rag_content:
                rag_content = f"Here are some examples that might help you:\n\n{current_variable.rag_content}"

            variation_id = str(uuid.uuid4())

            if completion_text:
                
                temp_variation_prompt = initial_variation_prompt.format(prompt=prompt_text,
                                                                            variable_name=name,
                                                                            completion_type="",
                                                                            completion=completion_text,
                                                                            start_with=start_with,
                                                                            last_values=last_values,
                                                                            rag_content=rag_content,
                                                                            note=note)
            else:
                
                temp_variation_prompt = initial_variation_prompt.format(
                                                        variable_name=variable_id_string,
                                                        rag_content=rag_content,
                                                        start_with=start_with,
                                                        last_values=last_values,
                                                        note=note,
                                                        context=prompt_text)
            
            temp_variation_prompt = clean_string(temp_variation_prompt)

            if isinstance(current_model, OpenAIInstructModel) or isinstance(current_model, LlamaCPPModel):

                start_messages = temp_variation_prompt
            
            elif isinstance(current_model, OpenAIChatModel):

                start_messages = [
                    {"role": "system", "content": current_model.system_prompt},
                    {"role": "user", "content": temp_variation_prompt},
                ]   
                
            elif isinstance(current_model, MistralChatModel):

                start_messages = [ChatMessage(role="user", content=temp_variation_prompt)]
                
            else:

                raise ValueError("Unknow type of model")

            if current_variable.validator:

                count = 1

                while True:

                    if count > current_variable.validator.retry_number:

                        new_value = Variations(id=variation_id, parent_id=parent_id, value=generated_value, error_message=new_message, confidence_score=current_confidence_score)
                        current_variable.values[variation_id] = new_value
                        break
                    
                    generated_value = current_model.ask(messages=start_messages)

                    if isinstance(current_model, OpenAIChatModel):
                        current_confidence_score = current_model.confidence_scores
                    else: 
                        current_confidence_score = {} 

                    self.template.variables[variable_id_string].values[parent_id] = Variations(id=variation_id, parent_id=parent_id, value=generated_value, confidence_score=current_confidence_score)
                   
                    function_name = current_variable.validator.function_name
                    from_notebook = current_variable.validator.from_notebook
                    additional_parameters = current_variable.validator.additional_parameters

                    param_dict = {}

                    for param in additional_parameters:

                        param_dict[param] = self.template.variables[param].values[parent_id]
                        
                    isValid, new_message = function_to_call(function_name, from_notebook, param_dict)

                    if isValid:

                        new_value = Variations(id=variation_id, parent_id=parent_id, value=generated_value)

                        current_variable.values[variation_id] = new_value

                        break

                    else:

                        if isinstance(current_model, OpenAIInstructModel) or isinstance(current_model, LlamaCPPModel):

                            start_messages = f"{start_messages}\n\nAssistant:{generated_value}\n\nUser:{new_message}"

                        elif isinstance(current_model, OpenAIChatModel):

                            start_messages.append({"role": "assistant", "content": generated_value})
                            start_messages.append({"role": "user", "content": new_message})

                        elif isinstance(current_model, MistralChatModel):
                
                            start_messages.append(ChatMessage(role="assistant", content=generated_value))
                            start_messages.append(ChatMessage(role="user", content=new_message))
                           
                        else:

                            raise ValueError("Unknow type of model")

                        count = count + 1

            else:

                generated_value = current_model.ask(messages=start_messages)
                
                new_value = Variations(id=variation_id, parent_id=parent_id, value=generated_value, confidence_score=current_model.confidence_score)
                current_variable.values[variation_id] = new_value

            
            last_values_list.append(generated_value)

            # Create the desired string format if last_values_list is not empty
            if last_values_list:
                last_values = "You must generate a content value that is not similar to following values:\n'''" + "\n".join(last_values_list) + "\n'''"
            else:
                last_values = ""
            
            variations[variation_id] = new_value

        return variations

    def generate_evol_instruct_prompt(self, initial_prompt:str):

        evol_prompt_template = load_file(path="files/evol_instruct.txt")

        evol_instruct_prompt = evol_prompt_template.format(number_of_prompts=str(self.template.prompt_variation_number), prompt=initial_prompt)

        start_messages = [
                        {"role": "system", "content": "Answer as a valid JSON like {\"prompts\": [\"XXXX\", \"YYYY\"]}"},
                        {"role": "user", "content": evol_instruct_prompt},
                ]

        evol_instruct_model = OpenAIChatModel(model_name=ModelName.GPT_35_TURBO_CHAT.value)

        diversified_prompt_list = evol_instruct_model.ask(max_tokens=512,
                                                            temperature=1,
                                                            messages=start_messages,
                                                            json_mode=True)

        evol_instruct_generated_prompt_list = json.loads(diversified_prompt_list)["prompts"]

        return evol_instruct_generated_prompt_list


    def get_completion_error_message(self, params:Dict[str, Variable]):

        error_str = ""

        for id, param in params.items():

            if param.error_message:
                error_str = f"{error_str}\n{param.error_message}"

        return error_str.strip()

    def get_prompt_error_message(self, params:dict):

        error_str = ""

        for param in params:
            error_message = self.template.variables[param].error_message

            if error_message:
                error_str = f"{error_str}\n{error_message}"

        return error_str

    def generate_data(self, output_path):

        # Extracting structures and variables from the template
        prompt = self.template.prompt
        prompt_variables = self.extract_variable_from_string(prompt)
        prompt_fixed_variables = self.extract_variable_dict_from_string(text=self.template.prompt)

        completion = self.template.completion
        completion_variables = self.extract_variable_from_string(completion)
        completion_fixed_variables = self.extract_variable_dict_from_string(text=self.template.completion)

        save_as_csv = True

        result = []

        if len(prompt_variables) > 0:
            # Start the recursive generation process with an empty dictionary for current variations
            prompts_parameters = self.contextual_generation(prompt_text=prompt, variables=prompt_variables, current_variation_dict={}, fixed_variables=prompt_fixed_variables)

            for p_param in prompts_parameters:

                prompt_param = {}

                for variable_id_string, prompt_variation in p_param.items():
                    if prompt_variation.id:
                        parent_id = prompt_variation.parent_id
                    prompt_param[variable_id_string] = prompt_variation.value
                    prompt_param[f"error_message_{variable_id_string}"] = prompt_variation.error_message
                    prompt_param[f"confidence_{variable_id_string}"] = str(prompt_variation.confidence_score)

                initial_prompt = prompt.format(**prompt_param)

                prompt_list = [initial_prompt]

                if self.template.prompt_variation_number > 0:

                    prompt_list = self.generate_evol_instruct_prompt(initial_prompt=initial_prompt)

                for prompt_text in prompt_list[:max(self.template.prompt_variation_number,1)]:
              
                    completion_parameters = self.contextual_generation(prompt_text=prompt_text, 
                                                                       completion=completion, 
                                                                       variables=completion_variables,
                                                                       current_variation_dict={}, 
                                                                       fixed_variables=completion_fixed_variables,
                                                                       parent_id=parent_id)
                    
                    for c_param in completion_parameters:

                        completion_param = {}

                        for variable_id_string, variation in c_param.items():
                            completion_param[variable_id_string] = variation.value
                            completion_param[f"error_message_{variable_id_string}"] = variation.error_message
                            completion_param[f"confidence_{variable_id_string}"] = str(variation.confidence_score)

                        completion_result = completion.format(**completion_param)

                        if save_as_csv:

                            row = {"prompt": initial_prompt, "evol_prompt": prompt_text, "completion": completion_result}
                            row.update(prompt_param)
                            row.update(completion_param)
                            result.append(row)

                            write_to_csv(result, output_path)

        else:

            prompt_list = [prompt]

            for prompt_text in prompt_list[:max(self.template.prompt_variation_number,1)]:

                completion_parameters = self.contextual_generation(prompt_text=prompt_text, completion=completion, variables=completion_variables, current_variation_dict={}, fixed_variables=completion_fixed_variables)

                for param in completion_parameters:

                    completion_param = {}

                    for variable_id_string, variation in param.items():
                        completion_param[variable_id_string] = variation.value
                        completion_param[f"error_message_{variable_id_string}"] = variation.error_message

                    completion_result = completion.format(**completion_param)

                    if save_as_csv:

                        row = {"prompt": prompt, "evol_prompt": prompt_text, "completion": completion_result}
                        row.update(prompt_param)
                        row.update(completion_param)
                        result.append(row)

                        write_to_csv(result, output_path)


        return result
