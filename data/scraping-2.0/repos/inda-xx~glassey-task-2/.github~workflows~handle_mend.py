import os
import sys
import openai
import json

def main(key, file_path):
    openai.api_key = key
    
    file_name = file_path.split()[1]

    for root, dirs, files in os.walk(os.getcwd()):
        if file_name in files:
            file_path = os.path.join(root, file_name)
            break
    
    source_code = ""
    with open(file_path, 'r') as file:
        source_code = file.read()
        
    # Call openai api to generate question
    # See: https://platform.openai.com/docs/guides/chat/introduction for more information on the call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a teacher that wants to help a student by improving their code quality and code comprehensibility."},
            {"role": "assistant", "content": "Here is the code they want help with:"},
            {"role": "assistant", "content": source_code},
            {"role": "assistant", "content": "If the above code is not Java source code, then it is ok to tell them that the file they submitted was not a recognised Java file and they should try again."},
            {"role": "assistant", "content": "If the above code is Java source code, then give them a short list of code quality improvements that help improve code comprehensibility."},
            {"role": "assistant", "content": "Format the response in markdown format"},
        ]
    )
    fix_apostrophe = response.choices[0]['message']['content'].replace("\'","'")
    remove_quotes = json.dumps(fix_apostrophe)[1:-1]
    print(remove_quotes)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
