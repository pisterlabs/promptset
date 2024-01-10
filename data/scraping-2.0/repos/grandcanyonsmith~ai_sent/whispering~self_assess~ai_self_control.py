# create program that can do the following:
# 1.get user input for what they want to do
# 2. given the user input, do the following:
# 2a. create a temp_task.txt file where the ai will log the user input and all the code that the ai generates
# 2b. create a temp_code.txt file where the ai will log the code that the ai generates
# 2c. send a request to openai and ask it to generate a bash script to execute what the user wants
# 2d. create a temp_bash.sh file where the ai will log the bash commands that the openai generates
# 2e. execute the bash commands that the openai generates
# 2f. if it fails, append the error message to the temp_task.txt file, and then ask send the error message to openai and ask it to generate a bash script to fix the error
# Repeat 2c-2f until the ai can successfully execute the bash commands
# 2g. if it succeeds, append the success message to the temp_task.txt file
# 2h. notify the user that the ai has successfully executed the bash commands

# component 1: get_user_input
# component 2: create_temp_task_file
# component 3: create_temp_code_file
# component 4: ai_formatter_request (the ai that is responsible for formatting the user input and informing the supervisor ai what the user wants to do and what code the ai needs to generate based on whats in the temp_task.txt file)
# component 4: ai_supervisor_request (the ai that helps the ai_formatter_request to generate the code that the ai_formatter_request needs to generate based on whats in the temp_task.txt file)
# component 5: create_temp_bash_file
# component 6: execute_bash_commands
# component 7. log_analysis
# component 8. self_assess
# component 8. append_logs_to_task_file
# component 9. notify_user

import openai
import subprocess

def get_user_input():
    return input("What do you want to do? ").strip()

def create_temp_task_file(user_input):
    # create a temp_task.txt file where the ai will log the user input and all the code that the ai generates
    with open('temp_task.txt', 'w') as f:
        prompt = f"Hello, I am an AI that is responsible for helping my master to do the following: \n\n{user_input}\n\nHe said that if I asked you, you would be able to tell me the command to create a bash script that can do the task. Please help me by telling me the command to create a bash script that can do the task. Thank you.\n\nAI Supervisor: "
        f.write(prompt)
    return user_input

def create_temp_code_file():
    # create a temp_code.txt file where the ai will log the code that the ai generates
    with open('temp_code.txt', 'w') as f:
        f.write('')
    return None

def get_temp_task_file():
    with open('temp_task.txt', 'r') as f:
        temp_task_file = f.read()
    return temp_task_file

def ai_formatter_request(user_input):
    prompt = get_temp_task_file()

    # send a request to openai and ask it to generate a bash script to execute what the user wants
    openai.api_key = "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"
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n", "AI Formatter: "]
    )
    code = response.choices[0].text
    print("code: ", code)
    
    return code

def ai_supervisor_request(code):
    # send a request to openai and ask it to generate a bash script to fix the error
    openai.api_key = "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"
    response = openai.Completion.create(
    model="davinci",
    prompt=code,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n", "AI Supervisor: "]
    )
    code = response.choices[0].text
    return code

def create_temp_bash_file(code):
    # create a temp_bash.sh file where the ai will log the bash commands that the openai generates
    with open('temp_bash.sh', 'w') as f:
        f.write(code)
    return None

def execute_bash_commands():
    # execute the bash commands that the openai generates
    # run start a subprocess with the given arguments, and return True if return code is 0 (success), else return False
    # path cd ..
    path_to_script = './'
    path_to_script = path_to_script.rstrip()
    cmd = ['cd', path_to_script]
    
    subprocess.run(cmd)
    bash_script_path = 'self_assess/temp_bash.sh'
    bash_script_path = bash_script_path.rstrip()
    cmd = ['bash', bash_script_path]
    code = subprocess.run(cmd)
    print("code: ", code)
    return code

def log_analysis(run_success, code):
    if run_success.returncode == 0:
        return None
    with open('temp_task.txt', 'a') as f:
        f.write(f"\nError: AI Supervisor could not successfully execute bash_script.txt")
    code = ai_supervisor_request(code)
    print(code)
    return code

def append_logs_to_task_file(code):
    # append the success message to the temp_task.txt file
    if code is not None:
        with open('temp_task.txt', 'a') as f:
            f.write(f"\n{code}\n")
    else:
        with open('temp_task.txt', 'a') as f:
            f.write(f"\nSuccess!\n")

    return None

def notify_user():
    # notify the user that the ai has successfully executed the bash commands
    print("AI has successfully executed the bash commands")
    return None

def main():
    user_input = get_user_input()
    create_temp_task_file(user_input)
    create_temp_code_file()
    code = ai_formatter_request(user_input)
    
    run_success = execute_bash_commands()
    code = log_analysis(run_success, code)
    append_logs_to_task_file(code)
    notify_user()

    return None

if __name__ == '__main__':
    main()
