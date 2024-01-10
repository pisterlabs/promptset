import openai
import json
# Set your OpenAI API key 
api_path="C:/Users/FRENZY/Desktop/Github/apikey.txt"
with open(api_path, 'r') as file:
    # Read the first line
    openai.api_key = file.readline()

def extract_scenario(prompt):
    # Call the OpenAI API to generate scenario information from the prompt
    customized_prompt="Here is the prompt for a scenario of a phyics simulation: \n'"+prompt+ "'\nI want you to only return values in following format which will be fed as initial conditions (position, velocity and weight) to the simulator. You can add or remove number of particles, set their weights, their velocities(vx,vy,vx) and their positions(x,y,z) in a 3d space to best define scenario. It is better to keep a little distance between particles too and set their approach through velocity. Use the following format and do changes as mentioned: \n{\"particles\": [{\"x\": 1.0,\"y\": 0.0,\"z\": 0.0,\"vx\": 0.0,\"vy\": 0.0,\"vz\": 0.0,\"weight\": 1.0}, {\"x\": 1.0,\"y\": 0.0,\"z\": 0.0,\"vx\": 0.0,\"vy\": 0.0,\"vz\": 0.0,\"weight\": 1.0}]}"
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=customized_prompt,
        max_tokens=500,
        stop=None,
        temperature=0.7,
    )
    # Extract relevant information from the API response\
    parsed_scenario = parse_response(response.choices[0].text)
    print('----------------------------------')
    print(parsed_scenario)
    print('----------------------------------')
    return parsed_scenario



def parse_response(response_text):
     # Split the response text by newline to separate the scenario description and JSON object
    response_lines = response_text.strip().split('\n')

    # Extract the JSON object from the last line
    json_string = response_lines[-1]

    # Replace single quotes with double quotes in the JSON string
    json_string = json_string.replace("'", "\"")

    # Parse the JSON object
    data = json.loads(json_string)
    return data
