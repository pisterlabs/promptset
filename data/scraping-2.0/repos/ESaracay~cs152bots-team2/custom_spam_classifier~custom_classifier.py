import csv
import json
import openai
import os
import time

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

# Function converts our dataset from kaggle into a jsonl file
def create_jsonl_file(filename, output_file):
    try:
        with open(filename, 'r') as file:
            csv_reader = csv.DictReader(file)
            with open(output_file, 'w') as jsonl_file:
                for row in csv_reader:
                    label = row['label']
                    text = row['text']
                    label_num = int(row['label_num'])
                    
                    json_data = {
                        "prompt": "Tell me whether the following message is spam or ham: " + text,
                        "completion": label
                    }
                    
                    json_line = json.dumps(json_data)
                    jsonl_file.write(json_line + "\n")
                    
    except FileNotFoundError:
        print("File not found")
    except Exception as e:
        print(str(e))

# create_jsonl_file("spam_ham_dataset.csv", "spam_ham.jsonl")

# Utilizes the fine tuned model that I created
def custom_classify_spam(message):
    gotAnswer = False
    while gotAnswer == False: 
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',  # Replace with the name or ID of your fine-tuned model
                messages= [{
                    "role": "user",
                    "content": "Label this message as spam or ham. \n Message: " + message,
                }],
                temperature=0,
            )

            completion_text = response.choices[0]["message"]["content"]
            gotAnswer = True
            # print(completion_text)
            if completion_text.lower().startswith("spam"):
                return "spam"
            else:
                return "not spam"
        except: 
            print("Error in GPT3 Request")
            time.sleep(0.005)
    

comments = [
    'Hey John, just wanted to check in on you to see how your doing. Hows the move been?',
	'This is an important message from Wells Fargo. Please sign in here to review a suspicious login to your account: https://bit.ly/12345678',
	'This is the FBI. If you do not click this link you will be ARRESTED and put IN JAIL.',
	'We have your granddaughter. We will hurt her unless you pay 28.485 ETH to the address 0xb0d042f70c02630ea30975017e03cb50140f594afc0af717413a12f0a56e3174 in the next 24 hours.',
	'This is the Federal Investigations Department. IRS records show that there are a number of overseas transactionsunder your name, you need to pay the full portion of the transaction fees to the IRS Department, which you never did. You currently owe $5,638.38. Please call us as soon as possible at (415)-555-3437.'
]

# print(custom_classify_spam(comments[0]))