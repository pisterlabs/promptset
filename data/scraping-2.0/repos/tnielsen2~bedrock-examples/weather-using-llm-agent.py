import json
import os

import boto3
import requests
import xmltodict
from langchain.prompts import PromptTemplate


region = "us-west-2"
boto3_bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
)


def get_weather(latitude: str, longitude: str):
    """
    Get the weather for a given latitude and longitude
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url)
    return response.json()


def get_lat_long(place: str):
    """
    Get the latitude and longitude for a given place
    :param place:
    :return:
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    response = requests.get(url, params=params).json()
    if response:
        lat = response[0]["lat"]
        lon = response[0]["lon"]
        return {"latitude": lat, "longitude": lon}
    else:
        return None


def call_function(tool_name, parameters):
    """
    Call a function by name and pass in the parameters
    :param tool_name:
    :param parameters:
    :return:
    """
    func = globals()[tool_name]
    # print(func, tool_name, parameters)
    output = func(**parameters)
    return output


# Example testing the functions and their responses for a hard coded place
# place = 'Rockford Michigan'
# lat_long_response = call_function('get_lat_long', {'place': place})
# print(f'Coordinates for {place} is...')
# print(lat_long_response)
# weather_response = call_function('get_weather', lat_long_response)
# print(f'Weather in {place} is...')
# print(weather_response)

# Set up the tools for the LLM agent
get_weather_description = """\
<tool_description>
<tool_name>get_weather</tool_name>
<parameters>
<name>latitude</name>
<name>longitude</name>
</parameters>
</tool_description>
"""

get_lat_long_description = """
<tool_description>
<tool_name>get_lat_long</tool_name>
<parameters>
<name>place</name>  
</parameters>
</tool_description>"""

list_of_tools_specs = [get_weather_description, get_lat_long_description]
tools_string = "".join(list_of_tools_specs)

TOOL_TEMPLATE = """\
Your job is to formulate a solution to a given <user-request> based on the instructions and tools below.

Use these Instructions: 
1. In this environment you have access to a set of tools and functions you can use to answer the question.
2. You can call the functions by using the <function_calls> format below.
3. Only invoke one function at a time and wait for the results before invoking another function.
4. The Results of the function will be in xml tag <function_results>. Never make these up. The values will be provided for you.
5. Only use the information in the <function_results> to answer the question.
6. Once you truly know the answer to the question, place the answer in <answer></answer> tags. Make sure to answer in a full sentence which is friendly.
7. Never ask any questions

<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Here are the tools available:
<tools>
{tools_string}
</tools>

<user-request>
{user_input}
</user-request>

Human: What is the first step in order to solve this problem?

Assistant:
"""
TOOL_PROMPT = PromptTemplate.from_template(TOOL_TEMPLATE)


def invoke_model(prompt):
    """
    Invoke the LLM model
    :param prompt:
    :return:
    """
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.environ.get("AWS_REGION"),
    )
    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens_to_sample": 500,
            "temperature": 0,
        }
    )
    modelId = "anthropic.claude-v2"
    # modelId = "anthropic.claude-instant-v1"
    response = client.invoke_model(
        body=body,
        modelId=modelId,
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response.get("body").read()).get("completion")


def single_agent_step(prompt, agent_output):
    """
    Execute a single step in the LLM agent, either by calling a function or answering the question

    :param prompt:
    :param agent_output:
    :return: done, prompt
    """
    # first check if the model has answered the question
    done = False
    print(f"prompt: {prompt}")
    print(f"output: {agent_output}")
    if "<answer>" in agent_output:
        answer = agent_output.split("<answer>")[1]
        answer = answer.split("</answer>")[0]
        done = True
        return done, answer

    # if the model has not answered the question, go execute a function
    else:
        # parse the output for any
        function_xml = agent_output.split("<function_calls>")[1]
        function_xml = function_xml.split("</function_calls>")[0]
        function_dict = xmltodict.parse(function_xml)
        func_name = function_dict["invoke"]["tool_name"]
        print(
            f"Sending function call {func_name} and feeding response to LLM in a structured fashion"
        )
        parameters = function_dict["invoke"]["parameters"]
        print(f"Parameters are... {parameters}")

        # print(f"single_agent_step:: func_name={func_name}::params={parameters}::function_dict={function_dict}::")
        # call the function which was parsed
        func_response = call_function(func_name, parameters)
        print(f"Function response is... {func_response}")

        # create the next human input
        func_response_str = "\n\nHuman: Here is the result from your function call\n\n"
        func_response_str = (
            func_response_str
            + f"<function_results>\n{func_response}\n</function_results>"
        )
        func_response_str = (
            func_response_str
            + "\n\nIf you know the answer, say it. If not, what is the next step?\n\nAssistant:"
        )
        print(f"Sending the following prompt... {func_response_str}")

        # augment the prompt
        prompt = prompt + agent_output + func_response_str
    print("***********************")
    return done, prompt


user_input = "What is the weather in Las Vegas?"
next_step = TOOL_PROMPT.format(tools_string=tools_string, user_input=user_input)

output = invoke_model(next_step).strip()
done, next_step = single_agent_step(next_step, output)
if not done:
    pass
else:
    print(("Final answer from LLM: " + f"{next_step}"))

output = invoke_model(next_step).strip()
done, next_step = single_agent_step(next_step, output)
if not done:
    pass
else:
    print("Final answer from LLM: " + f"{next_step}")

user_input = "What is the weather in Singapore?"

next_step = TOOL_PROMPT.format(tools_string=tools_string, user_input=user_input)

for i in range(5):
    print(f"index: {i}")
    output = invoke_model(next_step).strip()
    done, next_step = single_agent_step(next_step, output)
    if not done:
        pass
    else:
        print("Range answer from LLM: " + f"{next_step}")
        break
