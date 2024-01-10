import os
import openai
import json
import sys
import datetime
from datetime import datetime
import random
import time

old_print = print

def timestamped_print(*args, **kwargs):
  old_print(datetime.now(), *args, **kwargs)

print = timestamped_print

def save_strings_to_py_file(solution_strings, folder_name="solution_pys"):
    os.makedirs(folder_name, exist_ok=True)

    # os.chdir(folder_name)
    all_files = os.listdir(folder_name)
    for f in all_files:
        os.remove(f)
    
    for i in range(len(solution_strings)):
        with open(f"{folder_name}/solution_{i}.py", "w") as fp:
            fp.write(solution_strings[i])

if __name__ == "__main__":

    #python3 run_edit_module.py example_output.json
    input_file_name = sys.argv[1]

    out_dir = os.path.dirname(input_file_name)

    #The following code is to save the example prompt and outputs
    with open(input_file_name,"r") as input_fp:
        data = json.load(input_fp)

        # with open('example_prompt.txt', 'w') as outfile:
            # json.dump(data["prompt"], outfile)
        
        save_strings_to_py_file(data, f"{out_dir}/first_pys")
    exit()


        

    sys.stdout = open(f'./out.log', 'w')
    run(input_file_name)
    sys.stdout.close()