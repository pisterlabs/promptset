import pandas as pd
import openai

# Set the OpenAI API key
openai.api_key = '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'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('Playground_Submission_Data.csv')

# Filter out incorrect rows and rows with model "FlanT5-XXL"
df = df[(df['correct'] == True) & (df['model'] != "FlanT5-XXL")]
df = df.reset_index()
df = df.sample(frac=1)
# randomize order

# Initialize a counter for matching outputs
match_count = 0
count = 0

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    try:
        # Determine which API call to make based on the model
        if row['model'] == 'gpt-3.5-turbo':
            response = openai.ChatCompletion.create(
                messages=[{"role": "user", "content": row['prompt']}],
                model=row['model'],
                temperature=0,
            )
            print(response['choices'][0]['message']["content"])
            # Check if the output matches the expected completion
            if response['choices'][0]['message']["content"].strip() == row['expected_completion']:
                match_count += 1
        else:
            response = openai.Completion.create(
                engine=row['model'],
                prompt=row['prompt'],
                max_tokens=1024,
                temperature=0,
                top_p=0,
                frequency_penalty=1,
                presence_penalty=1,
            )
            print(response["choices"][0]["text"])
            if response["choices"][0]["text"].strip() == row['expected_completion']:
                match_count += 1
    except Exception as e:
        print(f"An error occurred: {e}")
        continue
    count+=1
    print(match_count, count, row['prompt'])

print(f'The number of outputs that match their expected completions is: {match_count}')