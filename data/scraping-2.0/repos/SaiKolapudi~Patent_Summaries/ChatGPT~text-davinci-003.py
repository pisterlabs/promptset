import openai
import pandas as pd
import re
from tqdm import tqdm

# Set up your OpenAI API key
openai.api_key = "sk-"

# Load the DataFrame from the Excel sheet
df = pd.read_excel("All_Patent.xlsx")

# Create a new DataFrame to store the summaries
output_df = pd.DataFrame(columns=['Abstract', 'Claims', 'Summary'])

# Iterate over the rows in the DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Summaries"):
    # Get the abstract and claims
    abstract = row['Abstract']
    claims = row['Claims']

    # Clean the abstract and claims text
    abstract = re.sub(r'[^\x00-\x7F]+', '', str(abstract))
    claims = re.sub(r'[^\x00-\x7F]+', '', str(claims))

    # Combine the abstract and claims
    input_text = abstract + ' ' + claims

    # Check if the input text exceeds the maximum token limit
    if len(input_text) > 4097:
        # Make a request to the OpenAI API for generating the summary of the abstract
        response_abstract = openai.Completion.create(
            engine="text-davinci-003",
            prompt="You are a helpful assistant that generates summaries.\n\nText: " + abstract,
            max_tokens=2048,
            n=1,
            stop=None,
        )
        summary_abstract = response_abstract.choices[0].text.strip()

        # Make a request to the OpenAI API for generating the summary of the claims
        response_claims = openai.Completion.create(
            engine="text-davinci-003",
            prompt="You are a helpful assistant that generates summaries.\n\nText: " + claims,
            max_tokens=2048,
            n=1,
            stop=None,
        )
        summary_claims = response_claims.choices[0].text.strip()

        # Combine the summaries of the abstract and claims
        summary = summary_abstract + ' ' + summary_claims
    else:
        # Make a request to the OpenAI API for generating the summary
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="You are a helpful assistant that generates summaries.\n\nText: " + input_text,
            max_tokens=4097,
            n=1,
            stop=None,
        )
        summary = response.choices[0].text.strip()

    # Append the row to the output DataFrame
    output_df = output_df.append({'Abstract': abstract, 'Claims': claims, 'Summary': summary}, ignore_index=True)

# Save the output DataFrame to a new Excel file
output_df.to_excel("text-davinci-003.xlsx", index=False)


