import os
import subprocess
import openai
from dotenv import load_dotenv
import json
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv("config.env")

# Access the OpenAI key from the environment variable
openai.api_key = os.environ.get("OpenAiKey")
client = OpenAI()

functions = [
    {
        "name": "source_writer",
        "description": "generates python source code wrapped in double hashtags: ##.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_code": {
                    "type": "string",
                    "description": "generated source code wrapped in doulbe hashtags to preserve formatting."
                }
            }
        }
    }
]
def write_code(user_input):
    while True:
        try:
            import re
            print(f"User input: {user_input}")
            print("Writing that Code!!")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional python coder. Your skills are unmatched.  you are an autonomous python coder, so please do not use any 'Todo' style notations, just write the code. when you feel like using a 'todo' note, write the logic. "
                    },
                    {
                        "role": "user",
                        "content": f"use the following prompt to generate python source code in the form of a multi-line string {user_input}.  Make sure to provide complete scripts with full logic, this process will be automated, there won't be human intervention in the code writing process. you never generate double quotes '"' absolutely do not do that, ever. Your responses are returned in json format and double quotes crash everything. To reiterate, do not generate any '"' double quotations, also, please do not use backslashes or anything that might trigger an invalid control character json decoder error when the response is parsed later. if you need to use an apostrophe, put the text in regular quotation marks, do not use backslashes. Format your response so that it will be a valid json object, this means handling escapes and control characters properly so that the string you return can be properly loaded as a json object in python."
                    }
                ],
                functions=functions,
                function_call={
                    "name": functions[0]["name"]
                },
                max_tokens=10000
            )
            print(response)

            # Parse the response JSON string into a dictionary
            response_json = json.loads(response.choices[0].message.function_call.arguments, strict=False)#"choices"][0]["message"]["function_call"]["arguments"], strict=False)

            # Extract the source code
            source_code = response_json["source_code"]

            # Sanitize the source code to remove invalid control characters
            source_code = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F]', '', source_code)
            # source_code = source_code.replace("'''", '')

            # Print the sanitized source code
            print(f"Sanitized Source Code:\n{source_code}")
            return source_code
        except json.decoder.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("Retrying...")
            continue  # Retry the function if a JSONDecodeError occurs


def write_code_bakup(user_input):

    import re
    print(f"User input: {user_input}")
    print("Writing that Code!!")
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a useful command line assistant."
            },
            {
                "role": "user",
                "content": f"use the following prompt to generate python source code in the form of a multi-line string {user_input}.  Make sure to provide complete scripts with full logic, this process will be automated, there won't be human intervention in the code writing process. you never generate double quotes '"' absolutely do not do that, ever. Your responses are returned in json format and double quotes crash everything. To reiterate, do not generate any '"' double quotations, also, please do not use backslashes or anything that might trigger an invalid control character json decoder error when the response is parsed later. if you need to use an apostrophe, put the text in regular quotation marks, do not use backslashes. Format your response so that it will be a valid json object, this means handling escapes and control characters properly so that the string you return can be properly loaded as a json object in python."
            }
        ],
        functions=functions,
        function_call={
            "name": functions[0]["name"]
        },
        max_tokens=10000
    )
    print(response)

    # Parse the response JSON string into a dictionary
    response_json = json.loads(response["choices"][0]["message"]["function_call"]["arguments"], strict=False)

    # Extract the source code
    source_code = response_json["source_code"]

    # Sanitize the source code to remove invalid control characters
    source_code = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F]', '', source_code)
    #source_code = source_code.replace("'''", '')

    # Print the sanitized source code
    print(f"Sanitized Source Code:\n{source_code}")
    return source_code

def run_script(script_location):
    output = ""
    while True:
        with open(script_location, 'r') as file:
            content = f"{file.read()}"
        command = ["python", f"{script_location}"]

        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)

            print("Output of the new script:")
            print(output)
            lines = output.strip().split('\n')
            for line in lines:
                print("Line:", line)
            break  # If the script runs successfully, break the loop

        except subprocess.CalledProcessError as e:
            print("Error:", e.output)  # Print the error output if the command fails
            print("")
            print("Re-Writing Script")
            content = content + e.output
            rewrite = write_code(content)
            with open(script_location, 'w') as file:
                file.write(rewrite)
            print("Script re-written, trying again! ")
            print(output)
            continue  # If an error occurs, continue the loop to try again
    print("Success! :)")
    return output

def run_script_bak(script_location):
    with open(script_location, 'r') as file:
        content = f"{file.read()}"
    command = ["python", f"{script_location}"]

    # Run the command and capture the output
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
        # Now the 'output' variable contains the output of the other script
        print("Output of the other script:")
        print(output)

        # You can further process the output or use it as needed
        # For example, splitting it into lines:
        lines = output.strip().split('\n')
        for line in lines:
            print("Line:", line)


    except subprocess.CalledProcessError as e:
        print("Error:", e.output)  # Print the error output if the command fails
        content = content + e.output
        print("")
        print("re-writing script")
        rewrite = write_code(f"This is a script, and the error it generated, please rewrite it correcing the errors: {content}")
        with open(script_location, 'w') as file:
            file.write(rewrite)
        print("script re-writen, trying again")

def main(source_code, title):
    # Create the 'source_code' directory if it doesn't exist
    if not os.path.exists('source_code'):
        os.makedirs('source_code')

    # Generate a unique name for the source code file
    source_code_name = f'{title}.py'

    # Save the source code to the file
    source_code_file_path = os.path.join('source_code', source_code_name)
    with open(source_code_file_path, 'w') as file:
        file.write(source_code)

    print(f"Source code saved to: {source_code_file_path}")
    print("")
    print("Attempting to run script: ")
    print("")
    run_script(source_code_file_path)

# Example usage
if __name__ == "__main__":
    #user_input = "write a python script that generates a simple web page with a title, a header, and some content. have the user enter these variables via input() if there are any dependencies needed include a function to install them via subprocess call to pip and a call to that install function at the start of the script."
    #user_input = "write a breakout clone in python, it should be a complete clone, with a controllable 'character' bar that moves left to right with a and d respectively on the keyboard, there should be a ball that bounces around the screen and breaks blocks at the top of the screen when it collides with them. there should be 25 blocks randomly arranged among the top 3rd of the screen, and each block is worth 2 points, when all the blocks break the level is over. A new level with randomly arranged blocks will begin, the users score will continue to grow "
    #user_input = "lets write a python script with a gui, that will manage an excel file with the following headers: Title, blank, blank, System, Genre, Price Charting, Manual, Map. we want to be able to add entries, sort the entries by title header, and remove/edit entries."
    #user_input = "write a python script that prints 'Im the best' 10 times, and counts to 25"
    #user_input = "write a super mario brothers clone in python, using simple shapes, the player should be a cirlce, bad guys squares and triangles. "
    user_input = "write a breakout clone in python using pygame. There will be 25 randomly arranged blocks amongst the top third of the screen, the bricks should not touch each other and should be aligned on rows in coloumns, in random positions along those rows and coloumns. the paddle will move left and right. each block is worth 2 points, when all the blocks are broken the player advances to the next level. the bricks are once against randomly arranged, and the game continues. if the ball falls past the bottom of the screen the space bar can be used to reset it in motion from the center of the paddle. the ball should move at a normal pace, and get progressively faster each level. you are an autonomous python coder, so please do not use any 'Todo' style notations, just write the code. when you feel like using a 'todo' note, write the logic. "
    source_code = write_code(user_input)
    #source_code = source_code[2:-2]
    main(source_code, title="jumpman")