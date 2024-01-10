import os
import sys
import openai
import json

def main(key, files):
    openai.api_key = key
    for source_file in files:
        source_file = source_file.strip("./")
        source_file = source_file.strip()
        with open(source_file, 'r') as file:
            source_code = file.read()
        prompt = "Given the following student source code, generate a multiple choice question about the code to test understanding of the code. The question should have three answer options and explanations for each answer option.\n\n"
    
        # Call openai api to generate question
        # See: https://platform.openai.com/docs/guides/chat/introduction for more information on the call
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a teacher that wants to help a student understand their programming assignment."},
                {"role": "assistant", "content": prompt + source_code},
                {"role": "assistant", "content": "The response should be formatted as a json object with the following fields: question, answer1, answer2, answer3, explanation1, explanation2, explanation3."},
            ]
        )
        # Extract the question from the response
        question = response.choices[0]['message']['content']

        # Parse json object from response
        response_json = json.loads(question)

        # Open issue on the repository with the question
        title = "🤖 Answer this question about your code!"
        body = ("**Considering [`/" + source_file + "`](../blob/master/" + source_file + ")**" + " <br /> <br /> " +
            "**" + response_json['question'] + "**" + " <br /> <br /> " +
            "A: " + response_json['answer1'] + " <br /> " +
            "<details><summary>...</summary>" + "_Explanation: " + response_json['explanation1'] + "_" + " </details>" +
            "B: " + response_json['answer2'] + " <br /> " +
            "<details><summary>...</summary>" + "_Explanation: " + response_json['explanation2'] + "_" + " </details>" +
            "C: " + response_json['answer3'] + " <br /> " +
            "<details><summary>...</summary>" + "_Explanation: " + response_json['explanation3'] + "_" + " </details>")

        print(f"::set-output name=title::{title}")
        print(f"::set-output name=body::{body}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
 
