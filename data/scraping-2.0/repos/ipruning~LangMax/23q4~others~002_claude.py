import json
import os

import anthropic

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
model_index = "claude-v1.3"
client = anthropic.Client(api_key=ANTHROPIC_API_KEY)

context = f"{anthropic.HUMAN_PROMPT} How many toes do dogs have?{anthropic.AI_PROMPT}"
print(repr(context))

completion = client.completion(
    prompt=f"{anthropic.HUMAN_PROMPT} How many toes do dogs have?{anthropic.AI_PROMPT}",
    stop_sequences=[anthropic.HUMAN_PROMPT],
    model="claude-v1.3",
    max_tokens_to_sample=1000,
)["completion"]

print(repr(completion))

output_data = {"context": context, "completion": completion}

with open("output.json", "w") as json_file:
    json.dump(output_data, json_file)
