#!/opt/homebrew/bin/python3
#  <xbar.title>Copy-n-prompt</xbar.title>
#  <xbar.version>v0.2</xbar.version>
#  <xbar.author>Erik de Bruijn</xbar.author>
#  <xbar.author.github>ErikDeBruijn</xbar.author.github>
#  <xbar.desc>Performs ChatGPT prompts on the clipboard and returns the result on the paste buffer.</xbar.desc>
#  <xbar.image>http://www.hosted-somewhere/pluginimage</xbar.image>
#  <xbar.dependencies>python</xbar.dependencies>
#  <xbar.abouturl>https://erikdebruijn.nl/</xbar.abouturl>

import subprocess
import os
import sys
import yaml
import openai
import shlex
import json

print("Copy-n-prompt") # bar menu entry
print("---") # below is in the dropdown

def debug_to_file(debug_message, file_path="/tmp/xbar-gpt.log"):
    with open(file_path, "a") as f:
        f.write(debug_message + "\n")

# Read settings from the YAML file
try:
    settings_file = os.path.expanduser("~/.copy-n-prompt/.openai.yml")
    with open(settings_file, 'r') as f:
        settings = yaml.safe_load(f)
except:
    print("Couldn't open settings file with OpenAI API key.")
    sys.exit(0)

# Read prompts from the YAML file
try:
    prompts_file = os.path.expanduser("~/.copy-n-prompt/prompts.yml")
    with open(prompts_file, 'r') as f:
        prompts = yaml.safe_load(f)
except FileNotFoundError:
    print("Prompts YAML not found | color=red")
    print(f"Edit Prompts | bash=nano param1={prompts_file} terminal=true")
    sys.exit(0)

openai.api_key = settings.get('open_ai_api_key')

debug_to_file("Started copy-n-prompt!")

# Loop through the prompts and display them in the dropdown
for i, prompt in enumerate(prompts):
    print(f"{prompt['name']} | bash=/Users/erik/Dev/Copy-n-Prompt/copy-n-prompt.py param1={i} terminal=false")

editor = os.environ.get('EDITOR', "nano")
print("---")
print(f"Edit Prompts | bash={editor} param1={prompts_file} terminal=true")

# The actual GPT-4 call
if len(sys.argv) > 1:
    try:
        prompt_index = int(sys.argv[1])
        prompt_text = prompts[prompt_index]
    except (ValueError, IndexError):
        print("Invalid index")
        sys.exit(1)

    if prompt_text:
        clipboard_content = subprocess.getoutput("pbpaste")
        final_prompt = prompt_text['prompt'].replace("{paste}", clipboard_content)
        debug_to_file("Final prompt: " + final_prompt)
        escaped_prompt = json.dumps(final_prompt)[1:-1]
        subprocess.run(["/usr/bin/osascript", "-e",
                        f'display notification "Prompt: {escaped_prompt}" with title "Copy-n-Prompt"'])

        debug_to_file("PRE request")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            temperature=0.2,
            max_tokens=1000
        )
        try:
            gpt_response = response["choices"][0]["message"]["content"]
        except:
            debug_to_file("response ERROR!")
            debug_to_file("Reponse: " + str(response))
            print("ERROR")

        debug_to_file("gpt_response: " + str(gpt_response))

        subprocess.run("/usr/bin/pbcopy", universal_newlines=True, input=gpt_response)

        if prompt_text.get('alert', False):
            escaped_response = json.dumps(gpt_response)[1:-1]
            subprocess.run(["/usr/bin/osascript", "-e",
                            f'display notification "Ready to paste: {escaped_response}" with title "Copy-n-Prompt"'])
