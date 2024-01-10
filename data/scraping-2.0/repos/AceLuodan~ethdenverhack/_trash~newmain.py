import openai
import time
# from transformers import GPT2TokenizerFast

import dataset_loader as loader

from newsamples import samples
from newtest import test

KEY = 'key.txt'
prompt = samples + "\n" + test

# initialize the openAI API
key = loader.load_key(KEY)
openai.api_key = key

tries = 0
while True:
    try:
        result = openai.Completion.create(
            engine="text-ada-001",
            prompt=prompt,
            max_tokens=5
        )
        vulnerability = result["choices"][0]["text"]
        break
    except Exception:
        print(Exception)
        tries += 1
        if tries == 3:
            break

if "Yes" in vulnerability:
    print("Yes")
elif "No" in vulnerability:
    print("No")
else:
    print("Error", result)
