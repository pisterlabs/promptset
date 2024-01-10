import sys
import os
import openai
import json 

# gpt-translator.py 
# by Marc Alier 2023
# This script uses the OpenAI API to translate text from one language to another
# licensed under the GNU General Public License v3.0
# you need to have a key to use the OpenAI API
# you can get one here: https://beta.openai.com/docs/developer-quickstart/api-key
# this program looks for the key in a file stated in the mykeypath variable 

mykeypath='..//..//mykey.json'
## sample content for the file mykey.json
# {
#  "key": "copy_your_key_here"
# }


MODEL = "gpt-3.5-turbo"

def translate(text, origin_language, destination_language):
    query=f"Translate the following text from {origin_language} to {destination_language}, but ignore English words:\n {text}"
    print("enviant a openai")

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a useful translator.  You will respect markup code, timestamps, and similar code in your translations."
            },
            {"role": "user", "content": query},
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0,
    )
    print("enviant a respon")

    return response['choices'][0]['message']['content']

def main():
    with open(mykeypath, 'r') as f:
        data = json.load(f)
        print("OK")
     # Initialize the OpenAI API
    openai.api_key = data['key']
    
    
    # Get the text to translate from the command line
    if len(sys.argv) != 5:
        print("Usage: python gpt-translate.py text origin_language destination_language destination_file")
        sys.exit(1)
    filename = sys.argv[1]
    print(f"filename: {filename}")
    origin_language = sys.argv[2]
    print(f"origin_language: {origin_language}")
    destination_language = sys.argv[3]
    print(f"destination_language: {destination_language}")
    destination_file = sys.argv[4]
    print(f"destination_file: {destination_file}")
    file=open(filename, "r")
    text = [line.rstrip() for line in file]
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    # translate the text in blocks of 20 lines
    # and save the result in a destination_file    
    i=0
    print(text)
    while i < len(text):
        print(i)
        translated_text = translate("\n".join(text[i:i+20]), origin_language, destination_language)
        with open(destination_file, 'a') as f:
            f.write(translated_text)
        i += 20 

if __name__ == "__main__":
    main()