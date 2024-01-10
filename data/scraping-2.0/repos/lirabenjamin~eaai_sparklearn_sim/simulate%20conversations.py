import openai
import pandas as pd
import numpy as np
import concurrent.futures
import os
import datetime
import dotenv


dotenv.load_dotenv()
output_dir = "data/simulated_conversations5"
TEMP = 1

openai.api_key = os.getenv("OPENAI_KEY")
  
data = pd.read_csv("data/data50_w_correct.csv")

ids = data['UserId'].tolist()
prompts = essays = data['prompt'].tolist()

def simulate_conversation(id, prompt):
  response = openai.ChatCompletion.create(
      model="gpt-4",
      temperature=TEMP,
      messages=[
              {"role": "system", "content": prompt},
              {"role": "user", "content": "Generate one full conversation between the student and the tutor"},
          ]
  )
  result = response.choices[0].message.content
  now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  with open(f"{output_dir}/{id}_{now}_temp{TEMP}.txt", "w") as f:
    f.write(result)

simulate_conversation(ids[0], essays[0])

with concurrent.futures.ThreadPoolExecutor() as executor:
  executor.map(simulate_conversation, ids, essays)

def read_all_files_to_dataframe(directory):
    all_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.txt')]
    df_list = []

    for filename in all_files:
        with open(filename, 'r') as f:
            content = f.read()
            df_list.append({"filename": filename, "content": content})

    return pd.DataFrame(df_list)

combined_df = read_all_files_to_dataframe(output_dir)

combined_df.columns = ["id", "content"]
combined_df["id"] = combined_df["id"].str.replace("data/simulated_conversations5/", "")
combined_df["id"] = combined_df["id"].str.replace(".txt", "")
combined_df[["id", "timestamp", "temp"]] = combined_df["id"].str.split("_", expand=True)


combined_df.to_parquet("data/simulated_conversations5.parquet") 

# Now rate them for motivation

prompt = """
I will show you an exchange between a student learning math and an intelligent tutoring system. 
Your goal is to pay attention to what the student is saying, and estimate how this student is feeling with regards to five motivational states. Score them on a scale from 0 to 10.
Confidence: How confident is the student in their ability to solve the problem?
Frustration: How frustrated is the student with their learning experience?
Boredom: How bored is the student with their learning experience?
Curiosity/Interest: How interested/curious is the student about the topic?
Engagement: How engaged is the student with the learning experience?

Your response should be formatted as a python dictionary, with the five motivational states as keys, and the scores as values.
"""
output_dir = "data/conversations5_ratings"
def rate_conversation(id, prompt, conversation):
  response = openai.ChatCompletion.create(
    model="gpt-4",
    temperature=TEMP,
    messages=[
          {"role": "system", "content": prompt},
          {"role": "user", "content": f"Here is the conversation:\n{conversation}"},
          ]
  )
  result = response.choices[0].message.content
  now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  with open(f"{output_dir}/{id}_{now}_temp{TEMP}.txt", "w") as f:
    f.write(result)

conversations = combined_df["content"].tolist()
rate_conversation(ids[0], prompt, conversations[0])
with concurrent.futures.ThreadPoolExecutor() as executor:
  executor.map(rate_conversation, ids, [prompt] * len(ids), conversations)

combined_df = read_all_files_to_dataframe(output_dir)
combined_df.columns = ["id", "content"]
combined_df["id"] = combined_df["id"].str.replace("data/conversations5_ratings/", "")
combined_df["id"] = combined_df["id"].str.replace(".txt", "")
combined_df[["id", "timestamp", "temp"]] = combined_df["id"].str.split("_", expand=True)

# unroll the dictionary
import ast
combined_df["content"] = combined_df["content"].apply(ast.literal_eval)
df = pd.DataFrame(combined_df.content.tolist())

# combine df and combined_df
df = pd.concat([combined_df, df], axis=1)
df.to_parquet("data/conversations5_ratings.parquet")

# do ratings match ground truth?
df = pd.read_parquet("data/conversations5_ratings.parquet")
ground_truth = pd.read_csv("data/data50_w_correct.csv")[["UserId", "confidence", "frustration", "boredom", "curiosity", "engagement"]]

# join by id and UserId
df['id'] = df['id'].astype(int)
df = df.merge(ground_truth, left_on="id", right_on="UserId")

# get correlations between ground truth and ratings
df[["confidence", "frustration", "boredom", "curiosity", "engagement"]].corrwith(df[["Confidence", "Frustration", "Boredom", "Curiosity/Interest", "Engagement"]])

print(df[["confidence", "frustration", "boredom", "curiosity", "engagement"]].isnull().sum())
print(df[["Confidence", "Frustration", "Boredom", "Curiosity/Interest", "Engagement"]].isnull().sum())
print(df[["confidence", "frustration", "boredom", "curiosity", "engagement"]].nunique())
print(df[["Confidence", "Frustration", "Boredom", "Curiosity/Interest", "Engagement"]].nunique())
print((df[["confidence", "frustration", "boredom", "curiosity", "engagement"]].index == df[["Confidence", "Frustration", "Boredom", "Curiosity/Interest", "Engagement"]].index).all())

print(df["confidence"].corr(df["Confidence"]))
print(df["frustration"].corr(df["Frustration"]))
print(df["boredom"].corr(df["Boredom"]))
print(df["curiosity"].corr(df["Curiosity/Interest"]))
print(df["engagement"].corr(df["Engagement"]))

df[['confidence', 'frustration', 'boredom', 'curiosity', 'engagement','Confidence', 'Frustration', 'Boredom', 'Curiosity/Interest', 'Engagement']].corr()

df.to_csv("data/conversations5_gptratings_and_truth.csv", index=False)