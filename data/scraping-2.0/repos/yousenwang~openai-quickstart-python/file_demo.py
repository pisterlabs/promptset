import os
import json
import openai

from dotenv import load_dotenv # Add
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")



# with open('data.jsonl', 'r') as json_file:
#     json_list = list(json_file)

# for json_str in json_list:
#     result = json.loads(json_str)
#     print(f"result: {result}")
#     print(isinstance(result, dict))

# openai.File.create(
#   file=open("data.jsonl", "rb"),
#   purpose='fine-tune'
# )

res = openai.File.list()

json_object = json.dumps(res, indent=4)
with open("file_list.json", "w") as outfile:
    outfile.write(json_object)

res = openai.File.retrieve("file-K164kgEHnfwu9xpH39BxZ6Yg")

json_object = json.dumps(res, indent=4)
with open("file_retrieve.json", "w") as outfile:
    outfile.write(json_object)
"""
openai.error.InvalidRequestError: 
To help mitigate abuse, 
downloading of fine-tune training files is disabled for free accounts.
"""

# res = openai.File.download("file-wNtVrddnJeqAkmpy8WjYXyah")
# print(type(res))

"""
Analyzing...

- Your file contains 2 prompt-completion pairs. In general, we recommend having at least a few hundred examples. We've found 
that performance tends to linearly increase for every doubling of the number of examples
- All prompts end with suffix `?\nAgent:`
  WARNING: Some of your prompts contain the suffix `?
Agent:` more than once. We strongly suggest that you review your prompts and add a unique suffix
- All prompts start with prefix `Summary: You're a customer service chat bot.

Specific information: Customers are using our company's platform via web or mobile app.

###

Customer: `. Fine-tuning doesn't require the instruction specifying the task, or a few-shot example scenario. Most of the time you should only add the input data into the prompt, and the desired output into the completion
- All completions end with suffix `.\n`

Based on the analysis we will perform the following actions:
- [Recommended] Remove prefix `Summary: You're a customer service chat bot.

Specific information: Customers are using our company's platform via web or mobile app.
###

Customer: ` from all prompts [Y/n]: y


Your data will be written to a new JSONL file. Proceed [Y/n]: y

Wrote modified file to `.\data_prepared.jsonl`
Feel free to take a look!

Now use that file when fine-tuning:
> openai api fine_tunes.create -t ".\data_prepared.jsonl"

After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string `?\nAgent:` for the model to 
start generating completions, rather than continuing with the prompt. Make sure to include `stop=[".\n"]` so that the generated texts ends at the expected place.
Once your model starts training, it'll approximately take 2.47 minutes to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.
"""

