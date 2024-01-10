# the experiment ! get gpt4 in here
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import json
import copy 

llm = OpenAI(model_name="gpt-4")

preamble = """
lets play a game where you are transforming an input grid of numbers into an output grid of numbers

the numbers represent different colors:
0 = black
1 = blue
2 = red
3 = green
4 = yellow
5 = gray
6 = magenta
7 = orange
8 = cyan
9 = brown

"""
def nl_only_prompt(task):
    
    instruction = "here is the instruction of how to transform the grid: \n"
    instruction += task['description']['description_input'] + task['description']['description_output_grid_size'] + task['description']['description_output']
    
    input_grid = task['problem']['test'][0]['input']

    prompt = preamble + instruction + "\n\nThe input grid is:\n" + str(input_grid) + "\n\nWhat is the output grid?"
    return prompt 

def nl_and_io_prompt(task):
    
    instruction = "here is the instruction of how to transform the grid: \n"
    instruction += task['description']['description_input'] + task['description']['description_output_grid_size'] + task['description']['description_output']
    
    train_input = task['problem']['train'][0]['input']
    train_output = task['problem']['train'][0]['output']
    input_output_example = "\n\nhere is an example of an input grid and its corresponding output grid:\n"
    input_output_example += "example input grid:\n" + str(train_input) + "\nexample output grid:\n" + str(train_output) + "\n\n"

    input_grid = task['problem']['test'][0]['input']

    prompt = preamble + instruction + input_output_example + "\n\nThe input grid is:\n" + str(input_grid) + "\n\nWhat is the output grid?"
    return prompt 

def io_only_prompt(task):
    
    # instruction = "here is the instruction of how to transform the grid: \n"
    # instruction += task['description']['description_input'] + task['description']['description_output_grid_size'] + task['description']['description_output']
    
    input_output_example = "\n\nhere are examples of input grids and its corresponding output grids:\n"

    for example_id in range(len(task['problem']['train'])):
        train_input = task['problem']['train'][example_id]['input']
        train_output = task['problem']['train'][example_id]['output']

        input_output_example += "example input grid:\n" + str(train_input) + "\nexample output grid:\n" + str(train_output) + "\n\n"

    input_grid = task['problem']['test'][0]['input']

    prompt = preamble + input_output_example + "\n\nThe input grid is:\n" + str(input_grid) + "\n\nWhat is the output grid?"
    return prompt 

def nothing_prompt(task):
    
    # instruction = "here is the instruction of how to transform the grid: \n"
    # instruction += task['description']['description_input'] + task['description']['description_output_grid_size'] + task['description']['description_output']
    
    take_a_guess = "\n\n this is a very common grid pattern found in the ARC (abstraction and reasoning corpus). Even without telling the rule, you can try to take a best guess on what the output grid is:\n"

    input_grid = task['problem']['test'][0]['input']

    prompt = preamble + take_a_guess + "\n\nThe input grid is:\n" + str(input_grid) + "\n\nWhat is the output grid?"
    return prompt 

def io_to_inst_prompt(task):
    
    # instruction = "here is the instruction of how to transform the grid: \n"
    # instruction += task['description']['description_input'] + task['description']['description_output_grid_size'] + task['description']['description_output']
    
    input_output_example = "\n\nhere are examples of input grids and its corresponding output grids:\n"

    for example_id in range(len(task['problem']['train'])):
        train_input = task['problem']['train'][example_id]['input']
        train_output = task['problem']['train'][example_id]['output']

        input_output_example += "example input grid:\n" + str(train_input) + "\nexample output grid:\n" + str(train_output) + "\n\n"

    input_grid = task['problem']['test'][0]['input']

    prompt = preamble + input_output_example + "\n\nThe input grid is:\n" + str(input_grid) + "\n\nWhat is the output grid?"
    prompt += "\ngenerate the answer in 2 parts. first, using the given examples, write down the rule of how to transform the input grid into the output grid. second, apply the rule to the given input grid to get the output grid."
    return prompt 

if __name__ == '__main__':
    import random 

    # open results/larc_gpt4.json
    with open('results/larc_gpt4.json') as json_file:
        larc_gpt4 = json.load(json_file)

    print (len(larc_gpt4))
    
    full_keys = ['nl_only', 'nl_and_io', 'io_only', 'nothing', 'io_to_inst']
    for _ in range(10000000):
        # pick a random task
        r_id = random.randint(0, len(larc_gpt4)-1)
        task = larc_gpt4[r_id]
        if len(task['gpt4']) < len(full_keys):
            print (f"task {task['name']} has less than all the gpt4 responses, try to make it up")
            print (task['gpt4'].keys())
            full_keys_shuffle = copy.deepcopy(full_keys)
            random.shuffle(full_keys_shuffle)
            for key in full_keys_shuffle:
                if key not in task['gpt4']:
                    if key == 'nl_only':
                        prompt = nl_only_prompt(task)
                    elif key == 'nl_and_io':
                        prompt = nl_and_io_prompt(task)
                    elif key == 'io_only':
                        prompt = io_only_prompt(task)
                    elif key == 'nothing':
                        prompt = nothing_prompt(task)
                    elif key == 'io_to_inst':
                        prompt = io_to_inst_prompt(task)
                    
                    print (prompt)

                    if len(prompt) > 8000:
                        print ("skipping this one, too long > 8000")
                        continue

                    with get_openai_callback() as cb:
                        ans = llm(prompt)
                        print("tokens used ", cb.total_tokens)
                        print ("printing answer")
                        print (ans)
                        print ("done printing answer")

                    task['gpt4'][key] = ans

                    with open('results/larc_gpt4_newer.json', 'w') as outfile:
                        json.dump(larc_gpt4, outfile)







    # # load the successful_tasks_with_desciptor.json file
    # with open('successful_tasks_with_desciptor.json') as json_file:
    #     successful_tasks = json.load(json_file)
    # print (len(successful_tasks))
    
    # previous_results_task_name_description = set()

    # task_with_answers = []

    # for task in successful_tasks:
        
    #     # skip the task if we already have it
    #     if task['name'] + task['description']['description_input'] + task['description']['description_output_grid_size'] + task['description']['description_output'] in previous_results_task_name_description:
    #         print ("skipping this one, already have it")
    #         continue
        
    #     # add it to the seen set as well
    #     previous_results_task_name_description.add(task['name'] + task['description']['description_input'] + task['description']['description_output_grid_size'] + task['description']['description_output'])

    #     print ("number of tasks so far ", len(previous_results_task_name_description), "total ", len(successful_tasks))

    #     prompt = io_only_prompt(task)
    #     print ("prompt length ", len(prompt))
    #     if len(prompt) > 8000:
    #         print ("skipping this one, too long > 8000")
    #         continue
    #     print (prompt)
    #     print ("\n")
    #     # assert 0

    #     with get_openai_callback() as cb:
    #         ans = llm(prompt)
    #         print("tokens used ", cb.total_tokens)

    #     # ans = llm(prompt)
    #     # duplicate task the dictionary straight up and add the answer to it
    #     task_with_ans = copy.deepcopy(task)
    #     task_with_ans['gpt-4-response'] = ans
    #     task_with_answers.append(task_with_ans)

    #     print (ans)
    #     print ("\n\n\n")

    #     # dump the task_with_answers list to a json file, name it 'task_with_answers.json', in the direction /results
    #     with open('results/io_only_task_with_answers.json', 'w') as outfile:
    #         json.dump(task_with_answers, outfile)
