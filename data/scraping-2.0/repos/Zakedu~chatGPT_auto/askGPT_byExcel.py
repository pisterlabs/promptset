import pandas as pd
import openai

def askGPT(text):
    openai.api_key = "1234"
    response = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = text,
        temperature = 0.6,
        max_tokens = 150,
    )
    return response.choices[0].text

# read the prompts from the Excel file
df = pd.read_excel("prompts.xlsx", sheet_name="Sheet1")

# create an empty list to store the responses
responses = []

# iterate over the prompts and get the responses
for prompt in df["prompt"]:
    response = askGPT(prompt)
    responses.append(response)

# create a new DataFrame with the responses
output_df = pd.DataFrame({"prompt": df["prompt"], "response": responses})

# write the output DataFrame to an Excel file
output_df.to_excel("output.xlsx", index=False)
