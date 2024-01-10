import openai
import yaml
import pandas as pd
from time import sleep
with open(r"C:\Users\sha\key") as f:
    keys = yaml.safe_load(f)
openai.api_key = keys["openai"]



# new_input.write("wbalabadabeda")

input_df = pd.read_csv("customv3.5/input.csv", header=None)
input_df.columns = ['input']
output_df = pd.read_csv("customv3.5/output.csv", header=None)
output_df.columns = ['output']

input_list = input_df["input"].tolist()
output_list = output_df["output"].tolist()


while len(input_list)>0:
    _input_list = []
    _output_list = []
    for i in range(len(input_list)):
        new_input = open("customv3.5/new_input.txt", 'a')
        new_output = open("customv3.5/new_output.txt", 'a')
        print(i)
        _in = input_list[i]
        _out = output_list[i]
        print(_in, _out)
        new_input.write(_in+"\n")
        new_output.write(_out+"\n")
        try:
            result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant that can generate linux command example and its description."},
                    {"role": "user", "content": f"I will show you a description of a Linux command, can you use 5 different ways to say this using English? You can directly print out the 5 lines without saying anything else. {_in}"},
                ]
            )
            

            for i in result["choices"][0]["message"]["content"].splitlines():
                new_input.write(i[3:]+"\n")
                new_output.write(_out+"\n")
        except Exception:
            print(f"Failed during: {_in, _out}")
            _input_list.append(_in)
            _output_list.append(_out)
        # sleep(1)

        new_input.close()
        new_output.close()

    input_list = _input_list
    output_list = _output_list

