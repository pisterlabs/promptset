import json
import os
from bs4 import BeautifulSoup
import openai
import glob
import re

""" This script is used to filter the JSON files that contain the dialogues from the #help channels of the math discord server.
It uses GPT-4 to determine whether or not the problem being discussed in the dialogue involves the concept of 
the derivative from the Calculus 1 course (amongst other features of the problem and context). 
"""

messages = [
    {"role": "system", "content": "You are Calculus1GPT, a large language model whose expertise is reading through dialogues of students and tutors in the #help channels of a mathematics discord, and determining whether or not they are discussing problems pertaining to the derivative. "}
]

class Chatbot():
    
    def parse_json(self, json_data):
        # This function takes a JSON file and returns a list of the contents of the file as a string.
        
        # Get the contents of the file as a string
        with open(json_data, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        html_content = json_data['dialogs']  
        
        html_content = re.sub(r'<[^>]+>', '', html_content)

        messages = []

        # pattern to match <b> tag with text inside parentheses
        pattern1 = r"<b>.*?\((.*?)\):</b>"

        # pattern to match <p> tag with text inside parentheses
        pattern2 = r"<p><b>.*?\((.*?)\):</b> (.*?)</p>"

        # combine both patterns with the | operator
        pattern = f"{pattern1}|{pattern2}"

        # find all matches and append them to messages list
        for match in re.finditer(pattern, html_content):
            messages.append(match.group())
        # Prepare the text for GPT-3
        gpt_input = ""
        

        for message in messages:
            gpt_input += message

        print(f'gpt_input is {gpt_input}')
        return gpt_input

    def create_prompt(self, gpt_input):
        prompt = """
        The input is going to a dialogue that took place online in a discord channel devoted to helping students with their math problems.
        Determine whether or not the problem involves the concept of the derivative from the Calculus 1 course. Additionally, looking at the dialogue between the users,
        determine the contexts (i.e., representations) in which the derivative is being used: That is, (1) graphically, as a slope of a tangent line; (2) verbally, as rate of change; 
        (3) physically, as a velocity in kinematic situations; (4) symbolically, as the limit of the difference quotient, or (5) "using rules to take the derivative"(e.g., a context not listed here). Similarly,
        if the value of derivative is 1, determine whether the problem is being solved with actions that treat the derivative as a ratio (for example, calculating the derivative as the slope of a 
        secant line between two points on a graph), as a limit (for example, calculating the derivative as the limit of the difference quotient), as a function (for example, calculating the slope of the tangent line to the curve
        of the original function at any point on the domain of the function). For the statement of the problem being solved that you will fill in below, try to extract the problem statement from the dialogue.
        Your answer will be in the following format: 
        1. 'derivative:[0 or 1], 
        2. problem:[statement of the problem being solved], 
        3. operator:[ratio, limit, function] 
        4. representational context [0 (if not about the derivative) 1 or 2 or 3 or 4 or 5; given by the representations above if derivative is 1], 
        5. representing topic of the problem:[derivative or other topic], 
        6. explanation:[Rest of explanation]'.
        The rest of the explanation should provide a justification for your answer, for each of the three questions, and provide examples of the operations or actions that the users are using when trying to solve the problem. 
        
         """
        prompt += gpt_input
        print('Done creating prompt')
        return prompt

    def gpt(self, prompt):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        print('got API key')
        messages.append({"role": "user", "content": prompt})
        r = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        print(f'GPT-4 response: {r}')
        answer = r['choices'][0]['message']['content']
        response = {'answer': answer} #    "answer": response['answer'],
        return response

    def process_gpt_response(self, original_file_path, response):
        # This function takes the response from GPT-4 and creates a JSON file to connect the response to the original JSON file.
        # The JSON file will be named the similarly to the original JSON file, but with the suffix "_response.json". 
        # For example, if the original JSON file is named "example.json", the response JSON file will be named "example_response.json".
        # The JSON file will be saved in the same folder as the original JSON file, but in a folder called "responses".
        
        # Get the name of the original JSON file
        file_name = os.path.basename(original_file_path)
        # Get the path to the folder containing the original JSON file
        file_path = os.path.dirname(original_file_path)
        # Create the path to the folder where the response JSON file will be saved
        response_path = os.path.join(file_path, "responses")
        
        # Create the 'responses' folder if it does not exist
        os.makedirs(response_path, exist_ok=True)
        # Create the path to the response JSON file
        response_file = os.path.join(response_path, file_name)
        # Create the response JSON file
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=4)
        print("Response JSON file created.")
        return response_file
    

if __name__ == "__main__":
    chatbot = Chatbot()
    
    while True:
        print("Please enter the path to a JSON file or a folder containing JSON files (or type 'exit' to quit):")
        path = input().strip()
        
        if path.lower() == "exit":
            break
        
        if os.path.isfile(path):
            if path.lower().endswith('.json'):
                # Process the file
                gpt_input = chatbot.parse_json(path)
                # Create the prompt
                prompt = chatbot.create_prompt(gpt_input)
                # Send the prompt to GPT-4
                response = chatbot.gpt(prompt)
                #  Process the response
                response_json = chatbot.process_gpt_response(path, response)
                
            else:
                print("The provided path is not a JSON file. Please provide a valid JSON file or folder.")
        elif os.path.isdir(path):
            json_files = glob.glob(os.path.join(path, "*.json"))
            if json_files:
                for file_path in json_files:
                    # Process the file
                    gpt_input = chatbot.parse_json(file_path)
                    # Create the prompt
                    prompt = chatbot.create_prompt(gpt_input)
                    # Send the prompt to GPT-4
                    response = chatbot.gpt(prompt)
                    #  Process the response
                    response_json = chatbot.process_gpt_response(file_path, response)
            else:
                print("The provided folder does not contain any JSON files.")
        else:
            print("Invalid path. Please provide a valid JSON file or folder.")