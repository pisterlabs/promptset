import openai
import pandas as pd

# set your api key
openai.api_key = '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'

# read the csv file
data = pd.read_csv('input_data.csv')

def fill_empty_cells(data):
    for column in data.columns:
        for i in data[data[column].isna()].index:
            prompt_data = ', '.join([f"{col}: {data.loc[i, col]}" for col in data.columns if pd.notna(data.loc[i, col])]) + '\n'
            
            prompt = prompt_data + f'{column}:'
            
            response = openai.completions.create(
              engine="davinci-codex",
              prompt=prompt,
              temperature=0.5,
              max_tokens=60
            )
            
            data.loc[i, column] = response.choices[0].text.strip()
    
    return data

# fill the empty cells
new_data = fill_empty_cells(data)

# write back to csv
new_data.to_csv('completed_data.csv', index=False)