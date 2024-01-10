import openai,os,sys
import json
from dotenv import load_dotenv
import random
import time


load_dotenv()
openai.api_key = os.environ.get('OPENAI_KEY')
chat_completion = openai.ChatCompletion()

def ask_chat_gpt(prompt):
    answer = None
    messages = []
    for delay_secs in (2**x for x in range(0, 6)):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
                ]
        
            messages.append(
                {"role": "user", "content": prompt}
            )
            response = chat_completion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            answer = response.choices[0].message.content
            messages.append({"role": "assistant", "content": answer})
            break
        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue
    return answer, messages

def ask_chat_gpt_twice(prompt, prompt2):
    answer = None
    messages = []
    for delay_secs in (2**x for x in range(0, 6)):
        try:
            messages = [
                {"role": "system", "content": "Help me on Software Engineering Tasks"}
                ]
        
            messages.append(
                {"role": "user", "content": prompt}
            )
            response = chat_completion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            answer = response.choices[0].message.content
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": prompt2})
            response = chat_completion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            answer = response.choices[0].message.content
            break
        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue
    return answer, messages
    
        
def code_completion_random_cut():
    answers = list()
    #json_path = '/workspaces/chat-gpt-failures/datasets/galeras_prompting/code_completion_docstring_3k_control.json'
    json_path = '/workspaces/chat-gpt-failures/datasets/galeras_prompting/code_completion_docstring_3k_T1_deduped.json'
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        print(len(json_data))
        
        for data in json_data:
            predicted = {}
            prompt ={}
            #prompt = "'Complete the following a {} code, return only code and complete method {}'".format('Python', data['random_split'])
            #prompt = "Write a {} method that starts with ```{}``` , I need to complete this function. Remove comments, summary and descriptions.".format('Python', data['random_split'])
            #p_template = "Complete the following python method: ```{}```"
            p_template = "Remember you have a Python function named {}, the function starts with the following code {}. The description for the function is: {} "
            p_text = p_template.format(data['fun_name'], data['random_cut'],data['documentation']["docstring"].strip())
            prompt2 = "remove comments; remove summary; remove description; Return only the code"
            answer, messages= ask_chat_gpt_twice(p_text,prompt2) #Change this for simple call
            p_text += prompt2 
            prompt['template'] = ''.join([p_template,prompt2])
            prompt['p_n_words'] = len(p_text.split())
            prompt['n_whitespaces'] = p_text.count(' ')
            prompt['vocab_size'] = len(set(p_text.split()))
            if not answer:
                print("Answer for data id {} was not generated".format(data["id"]))
                continue
            predicted['prediction'] = answer
            predicted['n_words'] = len(answer.split())
            predicted['n_whitespaces'] = answer.count(' ')
            predicted['vocab_size'] = len(set(answer.split()))
            data['T2'] = {'prompt':prompt, 'predicted': predicted}
            #data['control']['predicted'] = predicted
            #data['T1']['predicted'] = predicted
            answers.append(data)
       
    return answers
    

def codeserchnet_summarization_code():
    answers = []
    with open('datasets/code_xglue/text-to-code/codesearchnet/python/valid.jsonl', 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        question = result[0]['docstring']
        answers.append(ask_chat_gpt(question))
        
def save(name, data):
    with open('/workspaces/chat-gpt-failures/datasets/galeras_prompting/{}.json'.format(name), 'w') as f:
        print("saving data")
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def main():
    result = code_completion_random_cut()
    save("code_completion_docstring_3k_T2_deduped",result)

if __name__ == "__main__":
    sys.exit(main())
    
