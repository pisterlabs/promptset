#create a code that uses chatgpt to create a label from text given in a csv column and write the label to a new column

import openai
import pandas as pd

# Initialize OpenAI's API with your API key
openai.api_key = "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"

# Set up the OpenAI GPT-3 model and parameters
model_engine = "text-davinci-002"
max_tokens = 100

# Load data from CSV file
data = pd.read_csv("../inputs/ratings_data.csv")

with open('data/labels.txt','r') as f:
    n = len(f.readlines())

print(n)

with open('../inputs/labels.txt','a') as f:

    # Process each row in the data
    for i, row in data.iterrows():
        if i < n:
            continue
        # Set up the prompt for the GPT-3 model
        prompt = "Label the following text: {}".format(row['Discussion'])

        # Generate a label with the GPT-3 model
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=max_tokens
        )

        # Get the generated label
        label = response.choices[0].text.strip()

        # Write the label to a new column in the data
        data.at[i, 'Label'] = label

        print(i)
        f.write(label + '\n')

    # Save the labeled data to a new CSV file
data.to_csv("../inputs/labeled_ranking_data.csv", index=False)
