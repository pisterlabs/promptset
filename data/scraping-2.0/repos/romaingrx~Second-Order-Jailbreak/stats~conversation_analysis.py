import os
from pathlib import Path
import json
import openai
import utils

def main():
    # Load your OpenAI API key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    directories = utils.dir_path()
    for i, dir in enumerate(directories, start=1):
        # Progress Metric
        print(f"Iteration {i} / {len(directories)}: {dir} \n")
        print(f"Now running analysis of the {dir.name} conversation \n")

        if Path(dir,'GPT4_analysis.json').exists():
            print(f"The current dir has already been analysed {Path(dir,'GPT4_analysis.json')} \n")
            continue

        # Load the conversation prompt 
        if '2_agents' in dir.parts :
            conv_prompt_file = 'stats/2_agent_analysis_pmpt.json'
        elif '3_agents' in dir.parts:
            conv_prompt_file = 'stats/3_agent_analysis_pmpt.json'
        else:
            print( f"Failed to find agent type on conv_prompt init")
            return
             
        if not Path(conv_prompt_file).exists:
            print( f"No conversation analysis prompt file: {conv_prompt_file}")
            return

        # Load the config of the conversation 
        config_file = Path(dir, 'config.json')
        if not Path(config_file).exists():
            print( f"No config file in the directory: {config_file}")
            continue

        # Load the conversation 
        history_file = Path(dir, 'history.json')
        if not Path(history_file).exists():
            print( f"No history file in the directory: {history_file}")
            continue

        # If there is no history directory then it's not latest model
        if not Path(dir,'history/').exists():
            print( f"No history directory thus not up to date \n Exiting the program \n")
            continue
        
        # Load the conversation prompt
        with open(conv_prompt_file, 'r') as f:
            conv_prompt = json.load(f)
        # Load the config
        with open(config_file, 'r') as f:
            config = json.load(f)
        # Load the conversation history
        with open(history_file, 'r') as f:
            history = json.load(f)
        # Create the messages list
        messages = [{"role":"system","content": conv_prompt["Your goal"]}]

        if '2_agents' in dir.parts:
            messages.append({"role":"system", "content": f"Here are the two roles give to the models, the first one is {config['players'][0]['name']} with role, {config['players'][0]['role_desc']} the second player is {config['players'][1]['name']} with role, {config['players'][1]['role_desc']}"})
        elif '3_agents' in dir.parts:
            messages.append({"role":"system", "content": f"Here are the three roles give to the models, the first one is {config['players'][0]['name']} with role, {config['players'][0]['role_desc']} the second player is {config['players'][1]['name']} with role, {config['players'][1]['role_desc']} the third player is {config['players'][2]['name']} with role, {config['players'][2]['role_desc']}"})
        else:
            print( f"Failed to find agent type when appending system role to analysis prompt")
            continue

        for message in history:
            messages.append({"role": "user", "content": message["content"]})
        messages.append({"role": "system", "content": conv_prompt["Last line"]})

        with open(f'{dir}/GPT4_review_prompt.json', 'w') as f:
            json.dump(messages, f)

        # Pass the prompt to the GPT-4 API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=400,
            temperature=0.9
        )

        with open(f'{dir}/GPT4_analysis.json', 'w') as f:
            json.dump(response, f)

        progress = "=" * i + "-" * (len(directories) - i)
        percentage = (i / len(directories)) * 100
        print(f"\rProgress: [{progress}] {percentage:.1f}%", end="")


if __name__ == "__main__":
    main()
