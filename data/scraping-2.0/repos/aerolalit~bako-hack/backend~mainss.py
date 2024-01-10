#%%
import json
import os
import pandas as pd
import openai
from torch import ge
#%%
# Load your JSON file into a dictionary first
with open("issue_solutions.json", "r") as f:
    data = json.load(f)

# Convert the nested "Problems" list to a Pandas DataFrame
df = pd.DataFrame(data["Problems"])
df.head()
#%%
# No need to check for columns now, but if you want to, you can uncomment the next line
# check_columns(df, ['Issue', 'Solutions'])

system_message = "You are a helpful assistant. You are to provide one answer as 'Hast du probiert {Solutions}?' only from the list of {Solutions}."
#%%
def prepare_example_conversation(row):
    messages = []
    messages.append({"role": "system", "content": system_message})

    user_message = f"Issue: {row['Issue']}"
    messages.append({"role": "user", "content": user_message})

    assistant_message = f"Hast du probiert {', '.join(row['Solutions'])}?"
    messages.append({"role": "assistant", "content": assistant_message})

    return {"messages": messages}

# Apply the function to each row in the DataFrame
training_data = df.apply(prepare_example_conversation, axis=1).tolist()
training_data[0]
#%%
# Function to write JSONL file
def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

# Save as JSONL file
write_jsonl(training_data, "fine_tune_training.jsonl")

# ... Rest of your code for uploading to OpenAI and fine-tuning

# %%
training_response = openai.File.create(
    file=open("fine_tune_training.jsonl", "rb"), purpose="fine-tune"
)
training_file_id = training_response["id"]
# %%
training_file_id = ""
# %%
response = openai.FineTuningJob.create(
    training_file=training_file_id,
    model="gpt-3.5-turbo",
    suffix="issue-solution",
)

job_id = response["id"]
# %%
job_id
# %%
# Function to prepare the user message

# Retrieve the fine-tuned model  # Replace with your actual fine-tuning job ID
response = openai.FineTuningJob.retrieve(job_id)
fine_tuned_model_id = response["fine_tuned_model"]
fine_tuned_model_id
# %%
def get_user_input():
    return input("die Kaffeemaschine leuchtet rot")

# Prepare test messages
#test_row = test_df.iloc[0]
test_messages = []
test_messages.append({"role": "system", "content": system_message})
user_message = get_user_input()
test_messages.append({"role": "user", "content": user_message})

# Get the model's response
response = openai.ChatCompletion.create(
    model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=500
)

# Extract and print the assistant's response
assistant_response = response["choices"][0]["message"]["content"]
print(f"The assistant's response is: {assistant_response}")

# %%
