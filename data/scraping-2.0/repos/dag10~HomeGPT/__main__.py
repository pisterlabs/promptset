import requests
import json
import sys
import os
import openai
from string import Template


config = dict()

simulated_queries = dict(
        sim_on=('Turn on the island light',
                '[{"response":"Turning on the Island Light."},{"service":"light.turn_on","data":{"entity_id":["light.island_lights"]}}]'),
        sim_off=('Turn off the island light',
                 '[{"response":"Turning off the Island Light."},{"service":"light.turn_off","data":{"entity_id":["light.island_lights"]}}]'),
        sim_pink=('Turn the island light pink',
                  '[{"response":"Setting the island light to pink."},{"service":"light.turn_on","data":{"entity_id":["light.island_lights"],"color_name":"pink"}}]'),
        sim_lime=('Turn the island light lime',
                  '[{"response":"Setting the island light to lime."},{"service":"light.turn_on","data":{"entity_id":["light.island_lights"],"color_name":"lime"}}]'),
        sim_err1=('Generate an invalid service call',
                  '[{"service":"light.foobar","data":{"entity_id":["light.island_lights"],"color_name":"lime"}}]'),
        sim_err2=('Generate an invalid service call',
                  '[{"service":"light.turn_on","data":{}}]'),
        )
simulated_query = None


def load_config_or_exit():
    """Loads config.json."""
    global config

    try:
        config_file = open('config.json', 'r')
        config = json.load(config_file)
        config_file.close()
        # Access some values to throw exceptions if not found
        config['home_assistant_url']
        config['home_assistant_token']
        config['openai_api_key']
    except FileNotFoundError:
        print("Copy config.example.json to config.json and complete it.", file=sys.stderr)
        exit(1)
    except Exception as e:
        print("Invalid config.json: %s" % str(e), file=sys.stderr)
        exit(1)


def read_user_query_or_exit():
    """Load the next line of user input from stdin."""
    global simulated_query

    try:
        line = input("HomeGPT > ")
    except EOFError:
        exit(0)

    if line == "exit":
        exit(0)
    elif line == "demo":
        return "Which lights are on?"
    elif line in simulated_queries:
        simulated_query = simulated_queries[line]
        return simulated_query[0]

    return line;


def load_prompt_or_exit():
    """Loads prompt.txt."""

    try:
        prompt_file = open('prompt.txt', 'r')
        prompt_text = prompt_file.read()
        prompt_file.close()
        return prompt_text
    except FileNotFoundError:
        print("Couldn't find prompt file prompt.txt", file=sys.stderr)
        exit(1)
    except Exception as e:
        print("Failed to read prompt file prompt.txt: %s" % str(e), file=sys.stderr)
        exit(1)


def finish_initial_prompt(base_prompt):
    """Fill in the template variables in the prompt loaded from the file."""

    data = dict()
    return Template(base_prompt).substitute(data)


def execute_gpt_prompt(prompt):
    """Execute a finished prompt on GPT-3."""
    global simulated_query

    if simulated_query is not None:
        response_text = dict(choices=[dict(text=simulated_query[1])])
        simulated_query = None
        return response_text

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response


def print_sep(title=""):
    total_len = 50
    title_len = len(title)
    pad_front = round((total_len - title_len) / 2)
    pad_back = total_len - title_len - pad_front
    print(('=' * pad_front) + title + ('=' * pad_back))


def tell_user(text, error=False):
    """Tell the user some arbitrary plain text in response to something."""

    file = sys.stderr if error else sys.stdout
    print()
    print("HomeGPT says: " + text, file=file)


def execute_service(domain_and_service, data):
    """Call a Home Assistant service."""

    (domain, service) = domain_and_service.split('.')

    print()
    print_sep(" Calling service %s.%s " % (domain, service))
    print(data)
    print_sep()

    url = ('%s/api/services/%s/%s' % (config['home_assistant_url'], domain, service))
    auth_token = 'Bearer %s' % config['home_assistant_token']

    headers = {
            'Authorization': auth_token,
            'Content-Type': 'application/json'
            }
    body = data

    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        tell_user("Failed to call service %s.%s, got status code %s" % (domain, service, response.status_code),
                  error=True)
        return

    #print_sep(" Service call response (%s) " % response.status_code)
    #print(response.json())
    #print_sep()


def handle_raw_gpt_response(raw_response):
    """Handle raw GPT API response object. Returns GPT's response to the prompt."""

    try:
        response_text = raw_response['choices'][0]['text']
    except Exception as e:
        tell_user("Failed to parse text from raw GPT response: %s\n\nFull response is: %s" % (str(e), str(raw_response)),
                  error=True)
        return "[]"

    #print_sep(' GPT response text (unparsed) ')
    #print(response_text)
    #print_sep()
    #print()

    try:
        response_json = json.loads(response_text)
    except Exception as e:
        tell_user("Failed to parse GPT response JSON: %s" % str(e),
              error=True)
        return "[]"

    if not isinstance(response_json, list):
        tell_user("GPT response was not an array: %s" % str(e),
              error=True)
        return "[]"

    for response_obj in response_json:
        handle_gpt_response_obj(response_obj)

    print()

    return response_text


def handle_gpt_response_obj(response):
    """Handle a GPT response"""

    if 'response' in response:
       tell_user(response['response'])
    if 'service' in response:
        try:
            service = response['service']
            data = response['data']
        except Exception as e:
            tell_user("Response contained a malformed service call: %s" % str(e),
                  error=True)
            return
        execute_service(service, data)


if __name__ == '__main__':
    load_config_or_exit()
    base_prompt = load_prompt_or_exit()
    cumulative_prompt = finish_initial_prompt(base_prompt)

    openai.api_key = config['openai_api_key']

    while True:
        user_query = read_user_query_or_exit()
        cumulative_prompt += "Me:\n\"" + user_query + "\"\nComputer:\n"
        #print(cumulative_prompt)
        raw_response = execute_gpt_prompt(cumulative_prompt)
        response_text = handle_raw_gpt_response(raw_response)
        cumulative_prompt += response_text + "\n"

