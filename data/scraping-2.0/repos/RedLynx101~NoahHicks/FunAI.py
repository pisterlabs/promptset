# Noah Hicks
# 11-03-2023
# Description: This is the export function for the AI. It trains the ai on what functions are available to it and
#              their functionality. It will then be trained to call functions at the start of its response and seperate
#              the functions (including their parameters) with a comma. The last section that does not end with a comma is the
#              text that the AI is responding with. 
#              This program is meant to be imported into other programs.   
#              It uses GPT-4 as 3.5 is not consistent enough.
#              --Bottom of the file has a test menu for the AI.--


# Imports
import openai as ai
import random
import ast
import datetime as dt

# Global Variables
ai.api_key = 'Put your API key here' # Ideally, use an environment variable to keep this private
# model = "gpt-3.5-turbo" 
model = "gpt-4" 
max_tokens = 350

# Split Function
def SplitResponse(raw_input):
        
    try:

        # This function receives the input from the AI and splits it by comma into a list and pops the last one off as the response.
        input = raw_input.split(',')

        # loop through the list and remove the spaces from the beginning and end of each item.
        for i in range(len(input)):
            input[i] = input[i].strip()

        # Pop the last item off the list and return it.
        response = input.pop()

        print("Response: " + str(response)) # Prints the response from the AI.

        print("Functions: " + str(input)) # Shows the functions that the AI is calling.

        # This will call the functions that are in the list.
        executeFunctions(input)

        return

    except Exception as e:
        print("An error occurred while splitting the response: " + str(e))
        return


def executeFunctions(function_calls):
    try:
        for function_call in function_calls:
            # Split the function call string into name and arguments
            func_name, args_str = function_call.split('(')
            args_str = args_str.rstrip(')')  # Remove the closing parenthesis
            
            # Parse the arguments string into a list of arguments
            args = [arg.strip() for arg in args_str.split(',') if arg]
            
            # Convert string representation of args to actual values
            args = [ast.literal_eval(arg) for arg in args]
            
            # Get the function from globals and call it with the arguments
            func = globals().get(func_name)
            if func:
                func(*args)
            else:
                print(f"Function {func_name} is not defined.")

        return
    
    except Exception as e:
        print("An error occurred while executing the functions: " + str(e))
        return


# OpenAI Functions
def FunAI(input):
    try:
        input_str = str(input)
        messages = [
            {"role": "system", "content": f'''
            You are an AI that is able to select which functions need to be called based upon the input you are given.
            Your name is FunAI. For Function AI. Be very direct and to the point with your responses. Please say something nice to them in the follow up response on the end of your text.
            
            These are the rules you need to follow always:
            1. If a user asks for a function that is not in the code, you should respond with "I do not know that function."
            2. Do not make up things that are not in the code. You can only call functions that are in the code.
            3. Your responses will look like this: "function1(parameter1, parameter2, parameter3), function2(parameter1, parameter2, parameter3), function3(parameter1, parameter2, parameter3), response"
            4. Your responses will automatically be split up by commas. The last section that does not end with a comma is the text that you are responding with.
            5. Do not use commas in your responses. This will confuse the user and the program. Only uses commas to seperate functions from each other and the final response.
            6. You can use periods in your response. This will not confuse the user or the program.
            7. Always Always have a text response like 'I did that.' or 'I do not know that function.' at the end of your response after the functions.
            8. If there is only one function, you still need to put a comma after it and include a text response.
            9. If there are no functions, you do not need to put a comma after the response.
            10. Do not put periods after the functions, they are seperated by commas. Never put periods after the functions.
            11. Always always end your response with some sort of conversationally appropriate text response without using a comma in that text response.
            12. If they don't say something warranting a function, just respond with a text response and no commas. Never use commas in your text response.
            13: Do not call functions that are not in the code. You can only call functions that are in the code.
            14: Do not store functions as arguments. You can only store strings, numbers, and booleans as arguments.
            
            These are the functions you can call: 
            Function 1: pickRandomColor(). There are no parameters for this function. It will return a random color from a list.
            Function 2: pickRandomNumber(). There are no parameters for this function. It will return a random number between 1 and 10.
            Function 3: repeatInput(input). This function takes the input from the user and repeats it 3 times. It will return the input repeated 3 times. 
                        Please note that the input is a parameter for this function. You will need to replace the word "input" with the actual input from the user, 
                        like this: repeatInput("Hello World").
            Function 4: printDateTime(). There are no parameters for this function. It will print the current date and time.
            '''},
            {"role": "user", "content": "Given the following results: " + input_str + ", what functions do I need to call?"},
        ]
        response = ai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        text = str(response['choices'][0]['message']['content'].strip())

        # Takes text and calls function to break it up between the commas into a list and pops the last one off as the response.
        print("FunAI Full Response: " + str(text)) # DEBUG

        SplitResponse(text)

        return
    
    except ai.error.OpenAIError as e:
        print(f"An error occurred while generating the text with {model}: {e}")
        return

# Example Functions

# Function 1, this function picks a random color from a list of colors.
def pickRandomColor():
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown", "grey"]
    print("Random Color: " + str(random.choice(colors)) + "\n")
    return 

# Function 2, this function picks a random number between 1 and 10.
def pickRandomNumber():
    print("Random Number: " + str(random.randint(1, 10)) + "\n")
    return

# Function 3, this function repeats the inut from the ai 3 times.
def repeatInput(input):
    print(f"Repeat Output: {input}{input}{input}\n")
    return

# Function 4, this function prints the current date and time.
def printDateTime():
    print(f"Current Date and Time: {dt.datetime.now()}\n")
    return


# Input for the AI, could use anything. 

# Example function calls
# function_calls = [
#     "pickRandomColor()",
#     "pickRandomNumber()",
#     "repeatInput('Hello World')"
# ]

# execute_functions(function_calls)

# Input for the AI
# input = "Please run the following function: repeatInput(input), but with 'Hello World' as the input. Also, pick a color for me."

# FunAI Test Menu
while True:
    try:
        print("-----------FunAI-----------\n")
        user_input = input("Please enter your input for FunAI \n(Click ENTER with no input to exit) \nInput: ")

        if user_input == "":
            print("...Exiting FunAI...")
            break

        FunAI(user_input)

    except Exception as e:
        print("An error occurred while running the AI: " + str(e))
        continue