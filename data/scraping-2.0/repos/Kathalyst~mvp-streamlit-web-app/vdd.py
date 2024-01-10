import inspect
import llama2_process
import regex
import replicate
import vault
import os
import github_process

token = vault.get_Secret("replicate_key")
os.environ["REPLICATE_API_TOKEN"] = token

def identify_functions(file_names,file_contents):
    system_prompt = "Identify all defined functions in the input coding file. Return a python list with all the function names. Output should be in standard format 'filename: [function 1, function 2, etc.]'"
    functions = []
    results = []
    for i in range(len(file_names)):
        prompt = file_names[i] + ": \n" + file_contents[i]
        output = llama2_process.custom_prompt(system_prompt,prompt)
        print("\n\n\n")
        input = output + output
        matches = regex.extract_matches(input)
        print("Printing Matches from regex: ",matches)
        results.append(matches)
        #remove duplicates dictionaries from matches
        # for match in matches:
        #     functions.extend(match['functions'])
        # functions = list(dict.fromkeys(functions))
        # print("Printing functions: ",functions)
    return functions,results

def identify_functions_outside(file_names,file_contents):
    system_prompt = "Identify all functions in the input code that is imported or defined in another file. Return a python list with all the function names. Output should be in standard format 'filename: [function 1, function 2, etc.]'"
    functions = []
    results = []
    for i in range(len(file_names)):
        prompt = file_names[i] + ": \n" + file_contents[i]
        output = llama2_process.custom_prompt(system_prompt,prompt)
        print("\n\n\n")
        input = output + output
        matches = regex.extract_matches(input)
        print(matches)
        results.append(matches)
        #remove duplicates dictionaries from matches
        # for match in matches:
        #     functions.extend(match['functions'])
        # functions = list(dict.fromkeys(functions))
        # print(functions)
    return functions,results

def try_image_ai():
    prompt = '''

    I have a Python project with the following files:
    gpt.py:
import openai
import tiktoken
import vault
import file_data

openai.api_key = vault.get_Secret("openai_key")
multiprompt_flag = 0 # 0 for "No multi prompt" & 1 for "Yes multi prompt"
model = "gpt-4"
token_threshold = 7000
file_count_limit = 1

def create_base_prompt(file_contents,file_names):
    prompt = ""
    length = len(file_contents)
    # prompt = "I have a Python project with the following files: \n"
    for i in range(length):
        prompt = prompt + "\n" + file_names[i] + ": " + file_contents[i]
    # prompt = prompt + "\n Can you create a README file for this project?"
    return prompt

def check_prompt_tokens(prompt):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    print("Num of Tokens in prompt: ",num_tokens)
    if (num_tokens > token_threshold):
        print("Inside IF num tokens condition")
        print(num_tokens)
        print(token_threshold)
        multiprompt_flag = 1
    else:
        print(num_tokens)
        print(token_threshold)
        multiprompt_flag = 0
    return multiprompt_flag

def get_response_README(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
        {
            "role": "system",
            "content": "You will be provided with multiple file contents from one project, and your task is to create a README file for the project"
        },
        {
            "role": "user",
            "content": prompt
        }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return res

def parse_response(response):
    text = response['choices'][0]['message']['content']
    # print(text)
    return text

def control(file_contents,file_names,dir):
    base_prompt = create_base_prompt(file_contents,file_names)
    flag = check_prompt_tokens(base_prompt)

    print("Flag : ",flag)

    # if prompt tokens under limit, get response
    if (flag == 0): #No multiprompt
        print("\nNo multiprompt required.")
        res = get_response_README(base_prompt)
        with open(dir+'/prompt.txt','w+') as f:
            f.write(base_prompt)
        text = parse_response(res)
        with open(dir+'/README.md','w+') as f:
            f.write(text)
    # if prompt tokens not under limit, divide prompt and responses
    else: #Yes multiprompt
        total = len(file_names)
        text_response = []
        for i in range(0,total,file_count_limit):
            print("Inside FOR LOOP")
            base_prompt = create_base_prompt(file_contents[i:i+file_count_limit],file_names[i:i+file_count_limit])
            res = get_response_README(base_prompt)
            with open(dir+'/prompt'+str(i)+'.txt','w+') as f:
                f.write(base_prompt)
            text = parse_response(res)
            text_response.append(text)
            print("Text Response: ",text)
            with open(dir+'/README'+str(i)+'.md','w+') as f:
                f.write(text)
                print("Written")
        
        #combine all text_response values into one string variable
        text = ""
        for i in range(len(text_response)):
            text = text + text_response[i]
        with open(dir+'/README.md','w+') as f:
            f.write(text)
            print("Written")

    # return combined response
    return text

----------------------------------

Can you come up with a Flow diagram to show how the functions of the above files are related?'''

    output = replicate.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={
        "width": 768,
        "height": 768,
        "prompt": prompt,
        "scheduler": "K_EULER",
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "num_inference_steps": 50
    }
    )
    print(output)
    pass

def get_functions(file_path):
    with open(file_path, 'r') as file:
        print("Inside open")
        code = compile(file.read(), file_path, 'exec')
        print("Code: ",code)
        print(code.co_varnames)
        functions = [name for name, obj in inspect.getmembers(code) if inspect.isfunction(obj)]
        print("Functions: ",functions)
        return functions
    # Example usage
    pass

if __name__ == "__main__":
    
    github_link = "https://github.com/anushkasingh98/test-repo1.git"
    file_contents,file_names,dir = github_process.control(github_link)

    print("File Names\n",file_names)
    print("Count of Num of Files: ",len(file_contents))
    
    functions, matches = identify_functions(file_names,file_contents)

    print("First Round of Matches:\n",matches)

    functions1, matches1 = identify_functions_outside(file_names,file_contents)

    print(matches1)
