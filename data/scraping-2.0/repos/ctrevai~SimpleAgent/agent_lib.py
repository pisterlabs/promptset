from langchain.prompts import PromptTemplate
import requests
import boto3
import json
import xmltodict

# define tools and function


def get_weather(latitude: str, longitude: str):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url)
    return response.json()


def get_lat_long(place: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': place, 'format': 'json', 'limit': 1}
    response = requests.get(url, params=params).json()
    if response:
        lat = response[0]["lat"]
        lon = response[0]["lon"]
        return {"latitude": lat, "longitude": lon}
    else:
        return None


def call_function(tool_name, parameters):
    func = globals()[tool_name]
    # print(func, tool_name, parameters)
    output = func(**parameters)
    return output


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
tools_string = ''.join(list_of_tools_specs)

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
    client = boto3.client(service_name='bedrock-runtime',
                          region_name='us-east-1',)
    body = json.dumps(
        {"prompt": prompt, "max_tokens_to_sample": 500, "temperature": 0, })
    modelId = "anthropic.claude-v2:1"
    # modelId = "anthropic.claude-instant-v1"
    response = client.invoke_model(
        body=body, modelId=modelId, accept="application/json", contentType="application/json"
    )
    return json.loads(response.get("body").read()).get("completion")


def single_agent_step(prompt, output):

    # first check if the model has answered the question
    done = False
    if '<answer>' in output:
        answer = output.split('<answer>')[1]
        answer = answer.split('</answer>')[0]
        done = True
        return done, answer

    # if the model has not answered the question, go execute a function
    else:

        # parse the output for any
        function_xml = output.split('<function_calls>')[1]
        function_xml = function_xml.split('</function_calls>')[0]
        function_dict = xmltodict.parse(function_xml)
        func_name = function_dict['invoke']['tool_name']
        parameters = function_dict['invoke']['parameters']

        # print(f"single_agent_step:: func_name={func_name}::params={parameters}::function_dict={function_dict}::")
        # call the function which was parsed
        func_response = call_function(func_name, parameters)

        # create the next human input
        func_response_str = '\n\nHuman: Here is the result from your function call\n\n'
        func_response_str = func_response_str + \
            f'<function_results>\n{func_response}\n</function_results>'
        func_response_str = func_response_str + \
            '\n\nIf you know the answer, say it. If not, what is the next step?\n\nAssistant:'

        # augment the prompt
        prompt = prompt + output + func_response_str
    return done, prompt
