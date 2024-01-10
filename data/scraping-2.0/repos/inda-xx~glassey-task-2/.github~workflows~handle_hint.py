import os
import sys
import openai
import json

def main(key, file_path, exercise):
    openai.api_key = key

    with open(file_path, 'r') as file:
        assignment = file.read()
 
    exercise = exercise.split()[1]
    
    # Call openai api to generate question
    # See: https://platform.openai.com/docs/guides/chat/introduction for more information on the call
    debug = False
    if debug:
        sys.stdout.write("This is a test\nWith a new line")
    else:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a teacher that wants to help a student by giving hints and tips on a specific exercise. Here is their overall assignment:"},
                {"role": "assistant", "content": assignment},
                {"role": "assistant", "content": "Focus only on the specificed exercise " + exercise},
                {"role": "assistant", "content": "If the exercise does not exist, then it is ok to just say Sorry, but that exercise does not exist."},
                {"role": "assistant", "content": "If the exercise does exist, then give helpful hints and tips but do not provide a complete solution. Examples are good that help the student get started, such as 'You can try this...' or 'Think about this...' and so on."},
                {"role": "assistant", "content": "In terms of source code, at most you can provide skeleton code for inspiration, but it is the job of the student to write the code."},
                {"role": "assistant", "content": "Format the feedback as markdown format"},
            ]
        )
        
        fix_apostrophe = response.choices[0]['message']['content'].replace("\'","'")
        remove_quotes = json.dumps(fix_apostrophe)[1:-1]
        print(remove_quotes)
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
