import os
import sys
import openai
import json

def main(key, file_path):
    openai.api_key = key

    with open(file_path, 'r') as file:
        readme = file.read()
        
    # Call openai api to generate question
    # See: https://platform.openai.com/docs/guides/chat/introduction for more information on the call
    debug = False
    if debug:
        sys.stdout.write("This is a test\nWith a new line")
    else:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a teacher that wants to help a student by extending their programming task with a new exercise. Here is their overall task:"},
                {"role": "assistant", "content": readme},
                {"role": "assistant", "content": "Provide the next exercise that builds upon the original series of exercises and reinforces the skills they are testing."},
                {"role": "assistant", "content": "Do not repeat instructions already given in the overall task."},
                {"role": "assistant", "content": "Format the exercise in markdown and make sure it is a valid JSON string"},
            ]
        )
        fix_apostrophe = response.choices[0]['message']['content'].replace("\'","'")
        remove_quotes = json.dumps(fix_apostrophe)[1:-1]
        print(remove_quotes)
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
