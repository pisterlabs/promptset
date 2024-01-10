"""
Create an assistant using the tools from tool_creator using the assistant creation API
"""

import os
import json

from user_config import AssistantConfig as UserConfig 
from utils import chat as chat_loop

from openai import OpenAI
client = OpenAI() # be sure to set your OPENAI_API_KEY environment variable

def create_tool_user(assistant_details):
    # create the assistant
    tool_user = client.beta.assistants.create(**assistant_details["build_params"])

    print(f"Created assistant {tool_user.id} to use tools\n\n" + 90*"-" + "\n\n", flush=True)

    # save the assistant info to a json file
    info_to_export = {
        "assistant_id": tool_user.id,
        "assistant_details": assistant_details,
    }
    os.makedirs('assistants', exist_ok=True)
    with open('assistants/tool_user.json', 'w') as f:
        json.dump(info_to_export, f, indent=4)

    return tool_user

def talk_to_tool_user(assistant_details):
    """
    talk to the assistant to use the tools
    """

    # check if json file exists
    try:
        os.makedirs('assistants', exist_ok=True)
        with open('assistants/tool_user.json') as f:
            create_new = input(f'Assistant details found in tool_user.json. Create a new assistant? [y/N]')
            if create_new == 'y':
                raise Exception("User wants a new assistant")
            assistant_from_json = json.load(f)
            tool_user = client.beta.assistants.retrieve(assistant_from_json['assistant_id'])
            print(f"Loaded assistant details from tool_user.json\n\n" + 90*"-" + "\n\n", flush=True)
            print(f'Assistant {tool_user.id}:\n')
            assistant_details = assistant_from_json["assistant_details"]
    except:
        # create the assistant first 
        tool_user = create_tool_user(assistant_details)

    # gather the dependencies
    dependencies = assistant_details["dependencies"]
    if dependencies:
        print(f"Installing dependencies...", flush=True)
        for d in dependencies:
            os.system(f"pip install {d}")
        print(f"Installed dependencies\n\n" + 90*"-" + "\n\n", flush=True)

    # exec the functions from the py files
    os.makedirs('tools', exist_ok=True)
    functions = assistant_details["functions"]
    for func in functions:
        print(f"Loading function {func} into execution environment", flush=True)
        try:
            with open('tools/' + func + '.py') as f:
                exec(f.read(), globals())

            functions.update({func: eval(func)})
        except Exception as e:
            print(f"Exception loading function {func}: {e}", flush=True)
            print(f"Continuing without {func}...", flush=True)

    print(f"Loaded functions\n\n" + 90*"-" + "\n\n", flush=True)

    # Create thread
    thread = client.beta.threads.create()

    # chat with the assistant
    chat_loop(client, thread, tool_user, functions)

def main():
    # create the tool user assistant and chat to test your tools
    user_details = UserConfig().assistant_details
    talk_to_tool_user(user_details)


if __name__ == '__main__':
    main()