import openai
import yaml
import pandas as pd
from time import sleep
with open(r"C:\Users\sha\key") as f:
    keys = yaml.safe_load(f)
openai.api_key = keys["openai"]



# new_input.write("wbalabadabeda")

input_df = pd.read_csv(r"final_evalutaion\valid.csv", header=None, sep=' -> ')
input_df.columns = ['input', 'output']

input_list = input_df["input"].tolist()
output_list = input_df["output"].tolist()



for i in range(len(input_list)):
    output_df = open(r"final_evalutaion\altered_valid.txt", 'a')
    _in = input_list[i]
    _out = output_list[i]
    try:
        result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant that can generate linux command example and its description."},
                {"role": "user", "content": f"I will show you a description of a Linux command, can you use a different way to say this using English? Don't say anthing else except the altered description. {_in}"},
            ]
        )
        

        i = result["choices"][0]["message"]["content"]
        output_df.write(i + " -> " + _out + "\n")
        print(i + " -> " + _out)
    except Exception:
        print(f"Failed during: {_in, _out}")

    output_df.close()


