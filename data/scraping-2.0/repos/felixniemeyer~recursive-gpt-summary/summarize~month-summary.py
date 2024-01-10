import os 
import openai

from pathlib import Path

openai.api_key = "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"

basedir = str(Path.home()) + "/thoughts/"
startswith = "2023-03" # TODO: change to e.g. 2023-01

thoughts = ""
print("Including files:")
for filename in sorted(os.listdir(basedir)):
    if filename.startswith(startswith):
        with open(basedir + filename, "r") as f:
            print(filename)
            yyyy = filename[0:4]
            mm = filename[5:7]
            dd = filename[8:10]
            title = filename[11:-1]
            title = title.replace("-", " ")
            thoughts += yyyy + "-" + mm + "-" + dd + " " + title + ": \n"
            thoughts += f.read() + "\n\n"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Hi ChatGPT, Felix Niemeyer has accumulated reflections over the past days. He is thinking about topics in his mind and things that are happening to him in his live. Can you please bring that into a form like a book. Or a Chapter from a book. Improve the text where necessary, stick to the text when it's nice or witty words."},
        {"role": "user", "content": "please stick close to my words when possible. Don't summarize too much. Write from first person perspective."},
        {"role": "user", "content": thoughts},
    ]
)

print(response)

# write response to file
with open("./" + startswith + "-summary.txt", "w") as f:
    f.write(response["choices"][0]["message"]["content"])

