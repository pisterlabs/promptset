from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess
import random   
from colorama import Fore, init
from tqdm import tqdm

init(autoreset=True)


# Load the API key from the .env file
load_dotenv()

PROGRAMS_LIST = [
    '''Given two strings str1 and str2, prints all interleavings of the given
        two strings. You may assume that all characters in both strings are
        different.Input: str1 = "AB", str2 = "CD"
        Output:
        ABCD
        ACBD
        ACDB
        CABD
        CADB
        CDAB
        Input: str1 = "AB", str2 = "C"
        Output:
        ABC
        ACB
        CAB "''',
        "A program that checks if a number is a palindrome",
        "A program that finds the kth smallest element in a given binary search tree.",
        '''Develop a Python program that finds all the prime factors of a given number using an optimized algorithm. The program should take an integer as input and return a list of its prime factors. To make it more efficient, use the Sieve of Eratosthenes for finding prime numbers and then apply these primes to factorize the given number.''',
        "Write a Python program that merges two sorted linked lists into a single sorted linked list. The program should define a linked list structure, take two linked lists as input, and then combine them while maintaining the sorted order. The final output should be the merged sorted linked list."
    ]

def get_user_task(): 
    user_input = input("Tell me, which program would you like me to code for you? If you don't have an idea,just press enter and I will choose a random program to code: \n")
    if user_input.strip() == "":
        return random.choice(PROGRAMS_LIST)
    else :
        return user_input
        
    

def gen_code(user_input, client):
    # Initialize the OpenAI client
    prompt = "Python code only :" + user_input + " Do not write any explanation, comments, introduction or any other text besides the python code. Also please include complex unit tests using asserts method that check the logic of the program using 5 different inputs and expected outputs.Please print to the console the results of the unit tests. Once again, do not write any explanations, comments or introduction to this task too. "
           

    # Get the chat completion
    chat_msgs = [ {
                "role": "user",
                "content": prompt
            }]
    chat_completion = client.chat.completions.create(
        messages=chat_msgs,
        model="gpt-3.5-turbo",
    )

    # Extract the generated code from the response
    # Adjust the following line based on the actual structure of the ChatCompletion object
    generated_code = chat_completion.choices[0].message.content

    chat_msgs.append({
        "role": "assistant",
        "content": generated_code
    })
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path for the new file
    file_path = os.path.join(current_dir, 'userCode.py')

    # Write the generated code to the file
    with open(file_path, 'w') as file:
        file.write(generated_code)
    
    return chat_msgs

def run_and_fix_code(file_path, client, msgs=None, attempts=5):
    with tqdm(total=100, desc="Running and fixing code") as pbar:
        for attempt in range(attempts):
            try:
                result = subprocess.run(["python", file_path],check=True, capture_output=True, text=True)
                print(Fore.GREEN + ' Code creation completed successfully')
                pbar.update(100)  # Update progress bar to 100%
                cmd = f'start "" "{file_path}" ' 
                subprocess.call(cmd,shell=True) #This line works because of formatting to Windows style in previous line! Cannot work on MACOS or LINUX
                #os.startfile(file_path) #This line seems to open the file using the default app to open python code
                return
            except subprocess.CalledProcessError as e:
                print(Fore.YELLOW + f" Error running generated code! Error: {e.stderr}")
                pbar.update(100 / attempts)  # Update progress for each attempt
                error_message = f"There was an error in the generated code: {e.stderr}. Please fix the error without changing the purpose of the program. Once again, i want python only! Do not write any explanations, comments or introdution. Just write a new code, keeping the five unit tests that you wrote before, with the fixed error!"
                chat_msgs = (msgs or []) + [
                        {
                            "role": "user",
                            "content": error_message
                        }
                        ]
                
                chat_completion = client.chat.completions.create(
                    messages=chat_msgs,
                    model="gpt-3.5-turbo",
                )
                fixed_code = chat_completion.choices[0].message.content
                chat_msgs.append({
                    "role": "assistant",
                    "content": fixed_code
                })
                
                with open(file_path, 'w') as file:
                    file.write(fixed_code)

                if attempt == attempts - 1:
                    pbar.update(100) 

        print(Fore.RED + " Code generation FAILED")


if __name__ == '__main__':
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    user_input = get_user_task()
    msgs = gen_code(user_input,client)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'userCode.py')
    run_and_fix_code(file_path, client, msgs)
    
