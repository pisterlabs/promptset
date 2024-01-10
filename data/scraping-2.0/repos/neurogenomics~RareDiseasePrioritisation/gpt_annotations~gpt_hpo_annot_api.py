import openai
import os
import pandas as pd
import re
from IPython.display import display

# Initialize OpenAI client
client = openai.OpenAI(api_key="")

# Paths
input_csv = "~/Documents/gpt_prompts_extra.csv"
previous_outputs_file = '/Users/kittymurphy/Documents/gpt_hpo_annotations_extra.csv'

# Read data
prompts = pd.read_csv(input_csv)
outputs_df = pd.read_csv(previous_outputs_file)

# Initialize an empty DataFrame
df = pd.DataFrame()

# Get the last processed index or start from the beginning if none
last_processed_index = outputs_df.index.max() + 1 if not outputs_df.empty else 0
last_processed_index2 = (last_processed_index // 2) + 1

for index, row in prompts.iloc[last_processed_index2:].iterrows():
    prompt_content = row['prompt']

    try:
        response = client.chat.completions.create(
            model='gpt-4',
            temperature=0,
            messages=[
                {'role': 'user', 'content': prompt_content},
            ]
        )

        gpt_response = response.choices[0].message.content
        extracted_code = re.search(r'```python\n(.*?)\n```', gpt_response, re.DOTALL).group(1)
        exec(extracted_code)

        if isinstance(df, pd.DataFrame):
            if set(outputs_df.columns) == set(df.columns):
                df_sorted = df[outputs_df.columns]
                outputs_df = pd.concat([outputs_df, df_sorted], ignore_index=True)
                unique_df = outputs_df.drop_duplicates()
                unique_df.to_csv('/Users/kittymurphy/Documents/gpt_hpo_annotations_extra.csv', index=False)

    except Exception as e:
        print(f"An error occurred for prompt at index {index}: {e}. Skipping to the next iteration.")

    # Update last_processed_index with the current index + 1
    last_processed_index = index + 1
