import pandas as pd
import openai
import os


# This script reads in the java functions from the dataset
# and wraps them in the necessary code so they can be parsed by srcML
# in a later step.
# Uses calls to GPT model to accomplish this
# - running this took about 19 minutes, and cost 4 cents


# private
openai.api_key_path = "YOUR PATH HERE"

# dataset
csv = pd.read_csv("./pruned_seeds2.csv")
# redos = [31, 37, 90] # ChatGPT gave some description for these methods, so I had to manually redo them

# looping through all functions in dataset
for i in range(len(csv)):
#for i in redos:
    name = csv['name'][i]
    code = csv['function'][i]
    filename = f"wrapped_functions/{name}_wrapped.xml" # how the wrapped function will be saved

    # prompt to be sent to GPT model
    prompt = f"Please wrap this function in a java class so it can be parsed by srcML:{code}"

    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    
    # kind of a progress bar to see the functions print
    print(completion.choices[0].message.content)
    tree = completion.choices[0].message.content
    # creating new file
    with open(filename, "w") as f:
        f.write(tree)

