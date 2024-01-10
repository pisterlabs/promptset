import os
import sys
import openai
import json
import random

def main(key):
    openai.api_key = key
    
    # Strategy 2: Pick a random file from all java files found in repo
    source_file_list = [line.strip() for line in sys.stdin]
    source_file = random.choice(source_file_list)
    with open(source_file, 'r') as file:
        source_code = file.read()
        
    # Call openai api to generate question
    # See: https://platform.openai.com/docs/guides/chat/introduction for more information on the call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a teacher that wants to help a student understand their programming assignment."},
            {"role": "assistant", "content": "Given the following student source code, generate a multiple choice question about the code to test understanding of the code. The question should have three answer options and explanations for each answer option.\n\n"},
            {"role": "assistant", "content": source_code},
            {"role": "assistant", "content": "The response should be formatted as a json object with the following fields: question, answer1, answer2, answer3, explanation1, explanation2, explanation3."},
        ]
    )
    response_json = json.loads(response.choices[0]['message']['content'])

    # Print issue body
    print("**" + response_json['question'] + "**" + " <br /> <br /> " +  "A: " + response_json['answer1'] + " <br /> " + "<details><summary>...</summary>" + "_Explanation: " + response_json['explanation1'] + "_" + " </details>" + "B: " + response_json['answer2'] + " <br /> " + "<details><summary>...</summary>" + "_Explanation: " + response_json['explanation2'] + "_" + " </details>" + "C: " + response_json['answer3'] + " <br /> " + "<details><summary>...</summary>" + "_Explanation: " + response_json['explanation3'] + "_" + " </details>")

if __name__ == "__main__":
    main(sys.argv[1])
 
