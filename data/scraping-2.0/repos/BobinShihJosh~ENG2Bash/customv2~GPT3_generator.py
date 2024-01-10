import openai
import yaml
import pandas as pd

with open(r"C:\Users\sha\key") as f:
    keys = yaml.safe_load(f)

openai.api_key = keys["openai"]


def generate_command(command):
    prompt = f"""generate 50 differnt examples of bash command {command} and their desctiption with different folders."""

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.1,
        presence_penalty=0.1,
    )

    with open(command + ".txt", "w") as f:
        f.write(response.choices[0]["text"])


commands = pd.read_csv(r"C:\Users\sha\Desktop\ENG2BASH\finalized_commnads.csv")
for i in commands["commands"]:
    if i[0] != "-":
        print(i)
        generate_command(i.strip())
