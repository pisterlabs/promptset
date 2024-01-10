import openai
import pandas as pd
import re
from tqdm import tqdm
import time

# Set up your OpenAI API key
openai.api_key = "sk-"

# Load the DataFrame from the Excel sheet
df = pd.read_excel("All_Patent.xlsx")

# Create a new DataFrame to store the summaries
output_df = pd.DataFrame(columns=['Abstract', 'Claims', 'Abstract_Summary', 'Claims_Summary_1', 'Claims_Summary_2', 'Combined_Summary'])

# Iterate over the rows in the DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Summaries"):
    # Get the abstract and claims
    abstract = row['Abstract']
    claims = row['Claims']

    # Clean the abstract and claims text
    abstract = re.sub(r'[^\x00-\x7F]+', '', str(abstract))
    claims = re.sub(r'[^\x00-\x7F]+', '', str(claims))

    # Extract claims using regex pattern matching
    claim_list = re.findall(r"\d+\..+?(?=\n\d+\.|\Z)", claims, re.DOTALL)

    num_claims = len(claim_list)
    half_num_claims = num_claims // 2

    claims_1 = claim_list[:half_num_claims]
    claims_2 = claim_list[half_num_claims:]

    # If the number of claims is odd, add the extra claim to the second part
    if num_claims % 2 != 0:
        claims_2.append(claim_list[-1])

    # Make a request to the OpenAI API for generating the abstract summary
    abstract_response = None
    while abstract_response is None:
        try:
            abstract_response = openai.Completion.create(
                engine="text-davinci-003",
                prompt="You are a helpful assistant that generates abstract summaries.\n\nText: " + abstract,
                max_tokens=1000,
                n=1,
                stop=None,
            )
        except openai.error.ServiceUnavailableError:
            # Wait for 5 seconds before retrying the API request
            time.sleep(5)

    # Extract the generated abstract summary from the API response
    abstract_summary = abstract_response.choices[0].text.strip()

    # Make a request to the OpenAI API for generating the summaries for claim part 1
    claims_summary_1 = ""
    for claim in claims_1:
        claim_response = None
        while claim_response is None:
            try:
                claim_response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt="You are a helpful assistant that generates claim summaries.\n\nText: " + claim,
                    max_tokens=1000,
                    n=1,
                    stop=None,
                )
            except openai.error.ServiceUnavailableError:
                # Wait for 5 seconds before retrying the API request
                time.sleep(5)

        claim_summary = claim_response.choices[0].text.strip()
        claims_summary_1 += claim_summary + " "

    # Make a request to the OpenAI API for generating the summaries for claim part 2
    claims_summary_2 = ""
    for claim in claims_2:
        claim_response = None
        while claim_response is None:
            try:
                claim_response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt="You are a helpful assistant that generates claim summaries.\n\nText: " + claim,
                    max_tokens=1000,
                    n=1,
                    stop=None,
                )
            except openai.error.ServiceUnavailableError:
                # Wait for 5 seconds before retrying the API request
                time.sleep(5)

        claim_summary = claim_response.choices[0].text.strip()
        claims_summary_2 += claim_summary + " "

    # Combine the abstract, claims summaries, and claim parts summaries
    combined_summary = abstract_summary + ' ' + claims_summary_1 + ' ' + claims_summary_2

    # Make a request to the OpenAI API for generating the summary of combined_summary
    combined_summary_response = None
    while combined_summary_response is None:
        try:
            combined_summary_response = openai.Completion.create(
                engine="text-davinci-003",
                prompt="You are a helpful assistant that generates summaries.\n\nText: " + combined_summary,
                max_tokens=500,
                n=1,
                stop=None,
            )
        except openai.error.ServiceUnavailableError:
            # Wait for 5 seconds before retrying the API request
            time.sleep(5)

    # Extract the generated summary of combined_summary from the API response
    combined_summary_summary = combined_summary_response.choices[0].text.strip()

    # Append the row to the output DataFrame
    output_df = output_df.append({
        'Abstract': abstract,
        'Claims': claims,
        'Abstract_Summary': abstract_summary,
        'Claims_Summary_1': claims_summary_1,
        'Claims_Summary_2': claims_summary_2,
        'Combined_Summary': combined_summary_summary
    }, ignore_index=True)

# Save the output DataFrame to a new Excel file
output_df.to_excel("text-davinci-003.xlsx", index=False)
