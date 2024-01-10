import json
import openai  # for OpenAI API calls
import time  # for measuring time duration of API calls
import os
import re
import random
import pressgloss.core as PRESSGLOSS
import pressgloss.helpers as helpers

# Temporarily removing due to incompatibility with some systems 
# from transformers import T5ForConditionalGeneration


openai_model_list = [
    'curie',
    'text-davinci-003',
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4-32k'
    'babbage',
    'ada'
]
class gloss2daide: 
    def __init__(self, input=str, model=None, gloss=None, tones=None, tokenizer=None): 
        if tones is None:
            tones = random.sample(helpers.tonelist, random.randint(1, 3))
        else:
            self.tones = tones
        if model==None:
            model = 'gpt-3.5-turbo'
        # if model in openai_model_list:
        self.model = model
        openai.organization = os.getenv('OPENAI_ORG')
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key == None:
            print('Please set OPENAI_ORG and OPENAI_API_KEY environment variables')
            return

        if model in openai_model_list:
            if gloss == None: 
                utterance = PRESSGLOSS.PressUtterance(None, tones)
                english = ''.join(utterance.frompower) + ' ' + ' '.join(utterance.topowers) + utterance.english
                gloss = [{'role': 'user', 'content': english},
                                        {'role': 'assistant', 'content': utterance.daide}]
            while len(gloss) < 8:
                utterance = PRESSGLOSS.PressUtterance(None, tones)
                english = ''.join(utterance.frompower) + ' ' + ' '.join(utterance.topowers) + utterance.english

                gloss.extend([{'role': 'user', 'content': english},
                                        {'role': 'assistant', 'content': utterance.daide}])
            if model==[]: 
                print('No model specified, using default model')
                model='gpt-3.5-turbo'
            
            self.daide = self.build_chat_complete(gloss, input, model)
        else: 
            self.daide = self.finetune_completion_request(input, model)
        
        # else:
        #     self.daide = self.huggingface_translate(input, model, tokenizer)

    def finetune_completion_request(self, input, model):
         
        try:
            response = openai.Completion.create(
            model=model,
            prompt=input,
            max_tokens=100)
            content= response['choices'][0]['text']
            print("Before cleaner", content)
            content= helpers.grammar_cleaner(content)
            print("After cleaner", content)
            return response
        except Exception as e:
            print(f"Request failed due to {e}, trying again in 5 seconds")
            time.sleep(5)
    
    def build_chat_complete(self, gloss, input: str, model= 'gpt-3.5-turbo'):
        #This function uses a string to define a system and a list of dictionaries to define the tunning examples. 
        # start_time = time.time()
        #If the request fails, we will try again after 5 seconds

        message_data = [{"role": "system", "content": helpers.simple_system}]
        message_data.extend(gloss)
        message_data.append({"role": "user", "content": input})
        content =None 
        error = 'No_Error'
        while True:
                try:
                        response = openai.ChatCompletion.create(
                        model=model,
                        messages=message_data,
                        temperature=0)
                        content= response['choices'][0]['message']['content']
                        content= helpers.grammar_cleaner(content)
                        error = helpers.error_fetch(content)
                except Exception as e:
                        # print(f"Request failed due to {e}, trying again in 5 seconds")
                        time.sleep(5)

                
                
                break
        if error != 'No_Error':
                message_data.extend(
                        [{'role': 'assistant', 'content': content},
                        {"role": "user", "content": f"That's not correct DAIDE, {error}, try again"}])
                try:
                        response = openai.ChatCompletion.create(
                        model=model,
                        messages=message_data,
                        temperature=0)
                        content= response['choices'][0]['message']['content']
                        content= helpers.grammar_cleaner(re.sub('(.*)\\n\\n', '', content))
                        error = helpers.error_fetch(content)
                
                    

                except Exception as e:
                        print(f"Request failed due to {e}, trying again in 5 seconds")
                        time.sleep(5)
        if error != 'No_Error':
            return 'HUH?'
        else:
            return content
    # def huggingface_translate(self, input: str, model, tokenizer):
    #     input_ids = helpers.nlp_preprocess(input, tokenizer)
    #     model = T5ForConditionalGeneration.from_pretrained(model, device_map="auto")
    #     outputs = model.generate(input_ids)
    #     return helpers.decode_outputs(outputs, tokenizer)

class fine_tuned_model:
    def __init__(self, training_data=None, data_size=100000):
        openai.organization = os.getenv('OPENAI_ORG')
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if openai.api_key == None:
            print('Please set OPENAI_ORG and OPENAI_API_KEY environment variables')
            return
        self.training_list = []
        if training_data is None: 
            self.training_list = self.add_to_training_list(self.training_list, data_size)
            self.training_data = 'training_file'
        else: 
            if type(training_data) == list: 
                self.training_list = training_data
                self.training_data = 'training_file'
                helpers.dicts_to_jsonl(self.training_list, 'training_file')
            else: 
                self.training_data = training_data
        self.tracking_number = self.fine_tune_model(self.training_list, data_size)
            
        
    def fine_tune_model(self, training_list, n: int):
        training_list = self.add_to_training_list(training_list, n)
        helpers.dicts_to_jsonl(training_list, self.training_data)
        
        open_ai_feedback_cmd = f'yes | openai tools fine_tunes.prepare_data -f {self.training_data}.jsonl -q'
        while True:

            try:
                feedback = helpers.run_cmd(open_ai_feedback_cmd)
            except Exception as e:
                print(f"Request failed due to {e}, trying again in 5 seconds")
                time.sleep(5)
            break
        if 'error' in str(feedback['stdout']):
            print('Error in training data.')
            return feedback['stdout']
        

        tune_create_cmd = re.search(r'(?<=Now use that file when fine-tuning:\\n>\s)([\sa-zA-Z\:\-0-9.\_"()]+)', str(feedback['stdout'])).group(0)
            
        try:
            
            feedback = helpers.run_cmd(tune_create_cmd)
            print('finetune model feedback')
            print(feedback)
            model = re.search(r'(?<=Created\sfine-tune:\s)([a-zA-Z\-0-9.]+)', str(feedback['stdout']))
            while model is None:
                tracking_command = re.search(r'(?<=To\sresume\sthe stream,\srun:\\n\\n\s\s)([\sa-zA-Z\:\-0-9.\_]+)', str(feedback['stdout'])).group(0)
                feedback = helpers.run_cmd(tracking_command)
                print('tracking feedback')
                feedback = re.search(r'(?<=Created\sfine-tune:\s)([a-zA-Z\-0-9.]+)', str(feedback['stdout']))
            print('Model created, returning')
            model = str(model.group(0))
            print(model)
            return feedback
        except Exception as e:
            print(f"Request failed due to {e}, trying again in 5 seconds")
            time.sleep(5)
        
    def add_to_training_list(self, training_list, amount2add=int):
        if amount2add < 1: 
             return
        i = 0
        while i < amount2add:
            tones = random.sample(helpers.tonelist, random.randint(1, 3))
            utterance = PRESSGLOSS.PressUtterance(None, tones)
            english = ''.join(utterance.frompower) + ' ' + ' '.join(utterance.topowers) + utterance.english
            training_list.append({'prompt': english, "completion": utterance.daide})
            i += 1
        return training_list
    def fine_tune_predict(self, input: str)->str:

        if self.model == None: 
            fine_tune_list = helpers.run_cmd('openai api fine_tunes.list')
            if len(fine_tune_list) < 1:
                 self.fine_tune_model()
        else:
            res = gloss2daide.build_chat_complete(input, model=self.model)
            return res

         
class validate_model:
     def __init__(self, model, test_size, tones=None):
        self.model = model
        mismatch = 0
        parse_failure = 0
        i = 0
        if test_size is None: 
            test_size = 100
        
        if tones is None:
            tones = random.sample(helpers.tonelist, random.randint(1, 3))
        while i < test_size:
            utterance = PRESSGLOSS.PressUtterance(None, tones)
            english = ''.join(utterance.frompower) + ') ' + ' ('.join(utterance.topowers) + ') ' + utterance.english
            daide = utterance.daide
            encoding = gloss2daide(input=english, model=model)
            translation = encoding.daide
            if daide != translation:
                mismatch += 1
                try:
                    if helpers.error_fetch(translation) != 'No_Error':
                        parse_failure += 1
                    else:
                        pass
                except Exception as e:
                    print(f"API failure: {e}, probably server error, adding to failure rate.")
                    parse_failure += 1
            i += 1
        
        mismatch_rate = mismatch / test_size
        parse_failure_rate = parse_failure / test_size
        self.accuracy = (1 - mismatch_rate)*100
        self.parse_accuracy = (1 - parse_failure_rate)*100

            
        
