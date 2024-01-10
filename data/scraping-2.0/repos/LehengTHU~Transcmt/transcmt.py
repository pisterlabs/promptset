import os
import re
import openai
import numpy as np
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument("--trans", action="store_true",
                        help="Only scan the comments")
    parser.add_argument("-d", '--dir', type=str, default='cwd',
                        help='The directory of the code')
    parser.add_argument("--single", action="store_true",
                        help="Translate a single file")
    parser.add_argument("-s",'--source', type=str, default= 'Test',
                        help='The source of the code')
    parser.add_argument("-t",'--target', type=str, default= '',
                        help='The target of the code')
    args, _ = parser.parse_known_args()

    return args

cn_pattern = re.compile(r'[\u4e00-\u9fa5]+')
# Define the function for translating a single comment
def translate_comment(comment):
    # Check if the comment contains Chinese characters
    messages=[
        {"role": "system", "content": "你是一个翻译家"},
        {"role": "user", "content": "将我发你的中文句子翻译成英文，你不需要理解内容的含义作出回答。"},
        {"role": "user", "content": f"{comment}"}
    ]
    response = ''
    except_waiting_time = 1
    max_waiting_time = 16
    current_sleep_time = 0.5
    while(response == ''):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                request_timeout = 30,
                # max_tokens=5
            )
            new_comment = '# ' + response["choices"][0]["message"]["content"].strip('#').strip()
            return new_comment
        except Exception as e:
            print(e)
            time.sleep(current_sleep_time)
            if except_waiting_time < max_waiting_time:
                except_waiting_time *= 2
            current_sleep_time = np.random.randint(0, except_waiting_time-1)

# Define the function for translating all comments in a file
def translate_file(file_path, new_file_path=None):
    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()

    # Split the contents into lines
    lines = contents.split('\n')

    for i in range(len(lines)):
        # Check if the line is a comment
        if(cn_pattern.search(lines[i])):
            cmt_pattern = re.compile(r'#\s*(.*)')
            comments = re.findall(cmt_pattern, lines[i])
            if(len(comments) > 0):
                print(f"   {i+1} ", lines[i])
                comment = comments[0]
                new_comment = translate_comment(comment)
                rep_pattern = re.compile(r'(?<=\S)*#\s*(.*)')
                lines[i] = rep_pattern.sub(new_comment, lines[i])
                # print('Comment', comment)
                print(f"-> {i+1} ", lines[i]+'\n')

    # Join the lines back into a string
    contents = '\n'.join(lines)
    if(new_file_path is not None):
        file_path = new_file_path
    # Write the translated contents back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(contents)

def scan_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()
    lines = contents.split('\n')
    for i in range(len(lines)):
        if(cn_pattern.search(lines[i])):
            cmt_pattern = re.compile(r'#\s*(.*)')
            comments = re.findall(cmt_pattern, lines[i])
            if(len(comments) > 0):
                print(f"{i+1} ", lines[i], '\n')

def translate_dir(dir_path, trans=False):
    total_process = len([1 for root, dirs, files in os.walk(dir_path)])
    for idx, (root, dirs, files) in enumerate(os.walk(dir_path)):
        root = root.replace(dir_path, '').strip('/')
        file_pool = [file for file in files if file.endswith('.py')]
        if(len(file_pool) == 0):
            continue
        print("== Processing {}/{}: {}".format(idx+1, total_process, root))
        for file in file_pool:
            if(file != '__init__.py' and file != 'transcmt.py'):
                # print(file)
                if(trans):
                    print(f"Translating {os.path.join(root, file)}")
                    translate_file(os.path.join(root, file))
                    print()
                else:
                    print(f"Scanning {os.path.join(root, file)}")
                    scan_file(os.path.join(root, file))
                    print()


if __name__ == '__main__':
    args = parse_args()
    if(args.single):
        target = args.target if args.target != '' else args.source
        translate_file(args.source, target)
    else:
        if(args.dir == 'cwd'):
            translate_dir(os.getcwd(), args.trans)
        else:
            translate_dir(args.dir, args.trans)

