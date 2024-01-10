import os
from tqdm import tqdm
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


CLIENT = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
MODEL = "gpt-3.5-turbo"
OUTFILE = "old-interview.txt"
F = open(OUTFILE, "a")

def get_file_prefix(filenum):
    dir = ""
    if filenum <= 15:
        dir = "0-young/"
    elif filenum > 15 and filenum <= 24:
        dir = "1-falcon-captain/"
    elif filenum > 24 and filenum <= 32:
        dir = "2-new-hope/"
    elif filenum > 32 and filenum <= 58:
        dir = "3-resistance/"
    elif filenum > 58 and filenum <= 64:
        dir = "4-empire-rotj/"
    elif filenum > 64 and filenum <= 74:
        dir = "5-family-man/"
    elif filenum > 74:
        dir = "6-old/"
    return "data/hansolo/" + dir

def get_questions(summary):
    prompt = "Please generate 3 interview questions for the protagonist, Han, based on the following sequence of text:\n=====\n" + summary + "\n======\nPlease ask these questions directly to Han."
    questions_raw = CLIENT.chat.completions.create(
        model=MODEL,
        messages = [
            {
                "role": "system",
                "content": prompt
            }
        ],
        temperature=0
    )
    questions_list = questions_raw.choices[0].message.content.split("\n")
    for question in questions_list:
        F.write(question[3:].strip("\n") + "\n")
    

def main():
    for i in tqdm(range(66, 84)):
        filename = get_file_prefix(i) + f"hansolo-long-{i}.txt"
        f1 = open(filename, "r")
        f1.readline()
        summary = f1.readline()
        get_questions(summary)

if __name__ == "__main__":
    main()