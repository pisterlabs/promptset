import openai
import pandas as pd
import time

# Replace with your OpenRouter API key 
# enter your openrouter API KEY
OPENROUTER_API_KEY = "#####"

# OpenRouter API base URL
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = OPENROUTER_API_KEY

# Load the csv file containing contents in each posts
input_df = pd.read_csv("contents.csv")
input_df["label"] = None

# Use a for loop to send prompts via API and to get response from the LLM
for index, row in input_df.iterrows():
    if not pd.isna(row["label"]):
        print(f"{index} has been passed. (label has already existed)")
        continue
    index = row["index"]
    title = row["title"]
    content = row["content"]
    if pd.isna(title) or pd.isna(content):
        print(f"{index} has been passed. (title or content is none)")
        continue
    text = str(title) + " " + str(content)
    time.sleep(2)
    try:
        response = openai.ChatCompletion.create(
                    # choose a LLM model
                    model = "google/palm-2-chat-bison-32k",
                    # prompt sent to LLM model
                    messages=[
                    {"role": "system", "content": f"You are a language expert."}, 
                    {"role": "user", "content": f"Please help me to check if the following text contains sexually implicit or explicit content: {text}. Please be concise and only answer 'Yes' or 'No'. Do not elaborate."}
                    ]
                )
        # receive response from the LLM model
        api_response = response.choices[0].message["content"]
        print(f"#{index}")
        print(f"Title: {title}")
        print(f"Content: {content}")
        print(f"response: {api_response}")
        input_df.loc[input_df['index'] == index, 'label'] = api_response
    except:
        print(f"The row {index} fails to get the response from LLM")
    
    input_df.to_csv("labeled_data.csv", index= False)

input_df.to_csv("labeled_data.csv", index= False)

# use response to label posts
input_df['label_binary'] = input_df['label'].apply(lambda x: 1 if 'yes' in str(x).lower() else 0)

# Save the DataFrame to CSV
input_df.to_csv("labeled_data.csv", index=False, columns=["index", "link", "title", "content", "label", "label_binary"])
