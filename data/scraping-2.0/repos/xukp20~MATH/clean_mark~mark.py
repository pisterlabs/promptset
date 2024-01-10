"""
    GPT assisted marking
"""

SYSTEM_PROMPT = """
You will be given the names of a list of PDF files. I need your assistance in determining their category, level, and whether they're related to Mathematics based on their names.
Most of the files are related to Mathematics, but some are not.

Questions are:
1. Is the file related to Mathematics?
   - "yes" if it's related to Mathematics.
   - "no" if it's not related to Mathematics.
   - "unknown" if you're unsure about its relation to Mathematics.

2. Which field of Mathematics:
    - "ea" for Elementary Algebra
    - "ia" for Intermediate Algebra
    - "aa" for Advanced Algebra
    - "c" for Calculus
    - "g" for Geometry
    - "s" for Statistics
    - "t" for Trigonometry
    - "n" for not related to Mathematics, if you answered "no" to the previous question.
    - Other Fields (Specify, output a str in lower case)

3. Level of the file:
   - "ms" if it's suitable for middle school students.
   - "hs" if it's suitable for high school students.
   - "u" if it's suitable for undergraduate students.
   - "g" if it's suitable for graduate-level students.
   - "unknown" if you're unsure about the level or if it's not related to Mathematics.

4. Additionally, there are files of competitions, exams, textbooks, and workbooks.
   Please answer with one of the following:
   - "comp" if the file is related to competitions.
   - "exam" if the file is an exam.
   - "tb" if the file is a textbook.
   - "wb" if the file is a workbook.
   - "unknown" if you're unsure or if it's not related to Mathematics.

Give me the answers one line for each file, and divided by commas.
Here is an example:
----
File names: 
"(For Dummies  Math & Science) Steven Holzner - Physics Essentials For Dummies-Wiley (2010).pdf"
"IntroductoryStatistics-OP_i6tAI7e.pdf"
"Prepare to the SAT Subject Test (Math level 2).pdf"

Response:
no, n, unknown, unknown
yes, s, u, tb
yes, s, hs, exam
----
"""

USER_PROMPT = """
File names:
"{}"
"""

def get_messages(user_message):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

import os
import argparse
import json
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='Mark the books in an index.json')
    parser.add_argument('-i', '--input', type=str, help='input index file', default='index.json')
    parser.add_argument('-o', '--output', type=str, help='output index file', default='index.json')

    return parser

from openai_req import gpt3dot5_req
BATCH_SIZE = 1

def main():
    args = get_parser().parse_args()
    index_file = args.input
    output_file = args.output

    with open(index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)

    # get all file names
    # keep not rm ones or no rm tag ones
    batch_names = []
    batch_books = []
    for i, book in tqdm(enumerate(index), desc='Getting answers', total=len(index)):
        if book['clean'].get('rm', False):
            continue
        file_name = book['path']
        file_name = os.path.basename(file_name)

        batch_names.append(file_name)
        batch_books.append(i)

        if len(batch_names) == BATCH_SIZE:
            # get answers
            messages = get_messages(USER_PROMPT.format('\"\n\"'.join(batch_names)))
            
            tries = 0
            while tries < 3:
                try:
                    result = gpt3dot5_req(messages, temperature=0, max_tokens=300, top_p=1)
                except Exception as e:
                    print(e)
                    continue
                answers = result.split('\n')
                answers = [answer.split(',') for answer in answers]

                # test if the answers are valid
                if len(answers) == len(batch_names):
                    # check if all answers are valid
                    valid = True
                    for answer in answers:
                        if len(answer) != 4:
                            valid = False
                            print("Invalid answer: {}".format(answer))
                            break
                    
                    if valid:
                        # mark the books
                        for i, answer in zip(batch_books, answers):
                            index[i]['mark']['math'] = answer[0].strip()
                            index[i]['mark']['field'] = answer[1].strip()
                            index[i]['mark']['level'] = answer[2].strip()
                            index[i]['mark']['type'] = answer[3].strip()
                        # reset
                        batch_names = []
                        batch_books = []
                        break
                tries += 1
                messages.append({"role": "assistant", "content": result})
                messages.append({"role": "user", "content": "Invalid answers, please mention the format. New response:"})
                tqdm.write(str(len(batch_names)))
                tqdm.write(str(len(answers)))
                tqdm.write(messages[-3]['content'])
                tqdm.write(result)
                tqdm.write("Invalid answers, please try again.")


    # save index
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()