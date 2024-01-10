import openai
import pandas as pd
import requests
from tqdm import tqdm
import time

def score_title(title):
    message = {
        "role": "user",
        "content": f"Please rate the following article titles on a scale of -1 to 1, with higher scores being more positive and a score of 0 being emotionally neutral,you only need the score and no other information: \"{title}\""
    }

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[message]
    )
    # return response
    score_flt = float(response.choices[0].message.content.strip())
    return score_flt



api_key = "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" 
#czy's API For demonstration purposes only, please do not use this api key extensively.
openai.api_key = api_key
input_file = "your_output_csv_file.csv"
output_file = "your_output_csv_file.csv"

df = pd.read_csv(input_file)

# If the "scoregpt" column does not exist, create a new empty column
if "scoregpt" not in df.columns:
    df["scoregpt"] = None
i=0
y = 0
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing titles"):
    i+=1
    y+=1
    if y > 50:
        title = row["Title"]
        score = score_title(title)
        time.sleep(15) #Frequent requests will be actively rejected by the api
        df.at[idx, "scoregpt"] = score
    if i%5 == 0:
        df.to_csv(output_file, index=False)

df.to_csv(output_file, index=False)
print(f"Emotion scores have been saved to '{output_file}'.")
