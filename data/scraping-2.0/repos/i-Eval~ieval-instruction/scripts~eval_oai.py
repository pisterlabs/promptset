import time
import openai
from run import *
import os


"""
prepare: openAI 
"""

# config
openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
# openai.api_key = os.getenv("OPENAI_KEY")  # get openai_key via env var
openai.api_key = "" 
model_name = "text-davinci-002" 

def get_oai_completion(prompt, temperature=0.7, max_tokens=600, top_p=0.9):

    try: 
        response = openai.Completion.create(
            engine=model_name, 
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["<|EOS|>","<|im_end|>"]
        )
        gpt_output = response.choices[0].text
        return gpt_output
    except requests.exceptions.Timeout:
        # Handle the timeout error here
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.error.InvalidRequestError as e:
        # Handle the invalid request error here
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.error.APIError as e:
        if "The operation was timeout" in str(e):
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
            sleep(3)
            return get_oai_completion(prompt)            
        else:
            # Handle other API errors here
            print(f"The OpenAI API returned an error: {e}")
            return None


"""
prepare: prompt & evaluation
"""

# config
with open("./configs/alpaca.yaml", "r") as f:
    serving_config = yaml.safe_load(f)
PROMPT_DICT = serving_config["zero_shot_template"]
PROMPT_DICT


# load dataset
def get_ans_pair(datapath, debug=False):
    
    # load data
    task_name, dataset = load_dataset_and_batching(datapath, batch_size=10, few_shots=0) 
    # dataset: batch * samples * {instruction, input, output, few_shot_instances}
    if debug:
        print('---', task_name)

    # call GPT
    all_answers, all_gold_answers, cnt = [], [], 1
    cnt_max = 3 if debug else 99999999999
    for batch_i in dataset:
        for sample in batch_i:
            if len(sample['input']) != 0:
                final_prompt = PROMPT_DICT['with_input'].format_map({"instruction":sample['instruction'],"input":sample['input']})
            else:
                final_prompt = PROMPT_DICT['no_input'].format_map({"instruction":sample['instruction'],"input":sample['input']})

            success = False
            while not success:
                try:
                    ans = get_oai_completion(final_prompt, max_tokens=3)
                    success = True
                except:
                    time.sleep(1)
                    print('retry for sample:', cnt)

            all_answers.append(ans.strip())
            all_gold_answers.append(sample['output'])
            
            if debug:
                print('---input prompt:',final_prompt)
                print('--- ans from GPT:',ans)

            cnt += 1
            if cnt > cnt_max:
                break
            if cnt % 10 == 0:
                print(cnt)

        if cnt > cnt_max:
            break
            
    return task_name, all_answers, all_gold_answers


# evaluation
def metric_result(all_answers, all_gold_answers, save_to):
    
    # calculate accuracy
    accuracy = 0.
    total_number = 0.

    for answer, gold_answer in zip(all_answers, all_gold_answers):
        accuracy += matching_start_with(gold_answer, answer)
        total_number += 1
    accuracy /= total_number
    print("Accuracy: ", accuracy)
    
    # log modeloutput and accuracy
    with open(save_to, "w") as f:
        for answer, gold_answer in zip(all_answers, all_gold_answers):
            f.write(f"predicted:\t{answer} | gold:\t{gold_answer}\n")
        f.write(f"Accuracy: {accuracy}")
        
    return accuracy


"""
run 
"""
tasks, accs = [], []
debug = False
    
for fn in os.listdir("data/mmlu/"):
    print('----------',fn)
    datapath = "data/mmlu/"+fn

    task_name, all_answers, all_gold_answers = get_ans_pair(datapath, debug)
    save_to = f"results/ieval_log_{task_name}_{model_name}.txt"
    accuracy = metric_result(all_answers, all_gold_answers, save_to)
    tasks.append(task_name)
    accs.append(accuracy)
    for t, a in zip(tasks, accs):
        print(t, a)
    print('all avg:', sum(accs)/len(accs))