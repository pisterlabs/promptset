import sys
import os
import openai
import json 

# gramar-rev.py 
# by Marc Alier 2023
# This script uses the OpenAI API to make a gramatical and stype revision of a text
# it oututs a file with the revision, plus a folder with a visualizations of the revision 
# compared to the original text.
# licensed under the GNU General Public License v3.0
# you need to have a key to use the OpenAI API
# you can get one here: https://beta.openai.com/docs/developer-quickstart/api-key
# this program looks for the key in a file stated in the mykeypath variable 

mykeypath='..//..//mykey.json'
## sample content for the file mykey.json
# {
#  "key": "copy_your_key_here"
# }


# MODEL = "gpt-4" # this model is way more expensive

MODEL = "gpt-3.5-turbo"

def copy_edit(text):
    query=f"{text}"
    print("enviant a openai")

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a copy editor, you proofread and correct the text your receive. You make changes to the text to improve spelling, grammar, syntax and style. When you text, wrap the text in double tildes ~~,for inserted text, use markdown bold or italic to emphasize the new text. You will work in the language of the text. "
            },
            {"role": "user", "content": query},
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0,
    )
    print("Openai a respon")

    return response['choices'][0]['message']['content']

def main():
    with open(mykeypath, 'r') as f:
        data = json.load(f)
        print("OK")
     # Initialize the OpenAI API
    openai.api_key = data['key']
    
    
    # Get the text to translate from the command line
    if len(sys.argv) != 4:
        print("Usage: python grammar-rev.py origin-file  destination_file comparsion-folder")
        sys.exit(1)
    filename = sys.argv[1]
    print(f"filename: {filename}")
    destination_file = sys.argv[2]
    print(f"destination_file: {destination_file}")
    file=open(filename, "r")
    text = [line.rstrip() for line in file]
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    # Edit the text in blocks of 20 lines
    # and save the result in a destination_file    
    i=0
    print(text)
    while i < len(text):
        print(i)
        edited_text = copy_edit("\n".join(text[i:i+20]))
        with open(destination_file, 'a') as f:
            f.write(edited_text)
        i += 20 

if __name__ == "__main__":
    main()