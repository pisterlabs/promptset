import json
import os

import openai
import pandas as pd


def sms_to_email(i, sms):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a system that converts SMS messages into emails from Alice to Bob. Include a subject followed by a message in the emails you write.",
            },
            {"role": "user", "content": sms},
        ],
    )
    print(f"Finished generating email for SMS sample {i+1}")
    return response["choices"][0]["message"]["content"]


# Guardrail to make sure user actually wants to run this script
print(
    "Are you sure you want to run this script? It re-generates all the spam and non-spam email training data and uses OpenAI credits (provided by CS152)."
)

user_input = input("Enter `yes` to continue and anything else to quit: ")

if user_input.lower() != "yes":
    print("Exiting script...")
    quit()

# Load OpenAI organization and API key from 'tokens.json'
# There should be a file called 'tokens.json' inside the same folder as this file
token_path = "tokens.json"
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    tokens = json.load(f)
    openai.organization = tokens["openai_organization"]
    openai.api_key = tokens["openai_api_key"]

# Sample spam and non-spam entries from the Kaggle SMS spam dataset
NUM_SAMPLES_OF_EACH_CLASS = 500
df = pd.read_csv("kaggle_sms_spam.csv", encoding="ISO-8859-1")
sms_non_spam_sample = (
    df[df["v1"] == "ham"]
    .sample(n=NUM_SAMPLES_OF_EACH_CLASS, random_state=42)["v2"]
    .tolist()
)
sms_spam_sample = (
    df[df["v1"] == "spam"]
    .sample(n=NUM_SAMPLES_OF_EACH_CLASS, random_state=42)["v2"]
    .tolist()
)

# Save non-spam and spam emails to files
non_spam_directory = "non_spam_emails"
if not os.path.exists(non_spam_directory):
    os.makedirs(non_spam_directory)
spam_directory = "spam_emails"
if not os.path.exists(spam_directory):
    os.makedirs(spam_directory)

print("Generating non-spam emails...")
for i, sms in enumerate(sms_non_spam_sample):
    with open(
        os.path.join(non_spam_directory, f"non_spam_email_{i + 1}.txt"), "w"
    ) as f:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                f.write(sms_to_email(i, sms))
                break
            except Exception as e:
                print(
                    f"Failed to generate non_spam_email_{i + 1}.txt in attempt {attempt+1}, retrying... Error: ",
                    e,
                )
        else:
            print(
                f"Failed to generate non_spam_email_{i + 1}.txt after {max_retries} attempts."
            )

print("Generating spam emails...")
for i, sms in enumerate(sms_spam_sample):
    with open(os.path.join(spam_directory, f"spam_email_{i + 1}.txt"), "w") as f:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                f.write(sms_to_email(i, sms))
                break
            except Exception as e:
                print(
                    f"Failed to generate spam_email_{i + 1}.txt in attempt {attempt+1}, retrying... Error: ",
                    e,
                )
        else:
            print(
                f"Failed to generate spam_email_{i + 1}.txt after {max_retries} attempts."
            )
