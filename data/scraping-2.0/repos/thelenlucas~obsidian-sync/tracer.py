import os

from click import prompt
#Function that searches for a file in the director "Second Brain" and returns the path to the file (if it exists).
#Otherwise returns none.
def search_file(file_name):
    for root, dirs, files in os.walk("Second Brain"):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

# Path: tracer.py

#This function takes in a file name, scans through the file and returns a list of all the links in the file.
#Links are defined as words that start with [[ and end with ]].
def get_links(file_name):
    links = []
    with open(file_name) as f:
        for line in f:
            if "[[" in line:
                for word in line.split():
                    if "[[" in word:
                        links.append(word)

    #Strip the [[ and ]] from the links, and add Second Brain/ to the front of the links.
    #add the .md extension to the end of the links.
    for i in range(len(links)):
        links[i] = links[i].replace("[[", "")
        links[i] = links[i].replace("]]", "")
        links[i] = "Second Brain/" + links[i] + ".md"
        

    return links

def get_text_from_file(file_name):
    text = ""
    with open(file_name) as f:
        for line in f:
            text += line
    return text

def log(text):
    print(text)

#For a given file, this function grabs all the links, gets the text from the files that the links point to, and returns a string of all the text.
def get_text(file_name):
    links = get_links(file_name)
    text = ""
    for link in links:
        print(link)
        try:
            with open(link) as f:
                for line in f:
                    text += line
        except:
            log("One dead link")
    return text

i = input("Enter a subject:")

text = get_text("Second Brain/" + i + ".md")
question = "Explain the above subject (" + i +  ") to an undergraduate-level student:"
import os
import openai

openai.api_key = "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"
response = openai.Completion.create(
  model="text-davinci-002",
  prompt=text + "\n\"\"\"\n" + question,
  temperature=0.7,
  max_tokens=1901,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\"\"\""]
)

print(response.choices[0].text)