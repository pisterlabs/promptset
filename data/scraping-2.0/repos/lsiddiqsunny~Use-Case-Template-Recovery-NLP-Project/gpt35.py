import json
import openai


# %%
with open("./config.json") as f:
    config_data = json.loads(f.read())

OPENAI_KEY = config_data['OPENAI_KEY']
openai.api_key = OPENAI_KEY
print(OPENAI_KEY)


def gpt35_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt["code"]
                    + "\nGenerate a description of the code in this format:\nUse Case Description: <Description>\nInput Summary: <Input Summary>\nOutput Summary: <Output Summary>\nSolution:\n",
                }
            ],
            temperature=0.8,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=10,
        )
        prompt['output'] = response
        # print(response)
        return prompt
    except Exception as e:
        print(e)
        prompt['output'] = e
        return prompt
    

with open('./test_sample.json') as f:
    data = json.loads(f.read())

print(len(data))

new_data = []
for item in data:
    item = gpt35_response(item)
    new_data.append(item)
    break


# %%
with open('./GPT3.5_Output.json', "w") as f:
   json.dump(new_data, f, indent=4)