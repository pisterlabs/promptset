import openai
import yaml
import pandas as pd
with open(r"C:\Users\sha\key") as f:
    keys = yaml.safe_load(f)
openai.api_key = keys["openai"]





def generate_command(command):
    print(command)
    result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", 
            "content": "You are a helpful assistant that can generate linux command example and its description."},
        {"role": "user", 
            "content": f"Can you give me 100 examples of linux commands = \"{command}\" while following the rules I am giving you below?"},
        {"role": "assistant",
            "content": "Sure, what are the rules?"},
        {"role": "user",
            "content": "First, follow this format for output: {command} -> {description}"},
        {"role": "assistant",
            "content": "Sure, please tell me other rules."},
        {"role": "user",
            "content": "Second, you can use mock-up file/directory names. Each command should only include less than 5 words, with options separated."},
        {"role": "assistant",
            "content": "Sure, please tell me other rules."},
        {"role": "user",
            "content": "Finally, each descriptions should only talks about its function, options and directory"},
        {"role": "assistant",
            "content": "Sure, please tell me other rules."},
        {"role": "user",
            "content": "You can start outputing now."},
        ]
    )


    print(result["choices"][0]["message"]["content"])
    
    # with open(command + ".txt", "w") as f:
    #     f.write(result["choices"][0]["message"]["content"].replace("`", ""))



commands = pd.read_csv(r"C:\Users\sha\Desktop\ENG2BASH\finalized_commnads.csv")
for i in commands["commands"]:
    if i[0] != "-":
        generate_command(i.strip())
    break
