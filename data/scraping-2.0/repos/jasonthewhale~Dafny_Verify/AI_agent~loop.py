import re
import os
import json
import time
import openai
import subprocess

openai.api_key = 'API_KEY'

def turbo_completion(messages):
    completion = ""
    max_retry = 5
    retry = 0
    messages=messages
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            max_tokens=512,
        )
        reply = completion.choices[0].message['content']
        if '```dafny\n' in reply:
            code_start = reply.find('```dafny')
            code_end = reply.rfind('```')
            reply = reply[code_start+9:code_end]
        return reply
    except Exception as overload:
        retry += 1
        if retry >= max_retry:
            return "turbo error: %s" %overload
    

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()
    

def save_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)


def save_json(file_path, content):
    with open(file_path, 'w') as outfile:
        json.dump(content, outfile)


def verify(file_path, command):
    cli = command + file_path
    process = subprocess.Popen(cli, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    # Decode the output from bytes to string
    stdout = stdout.decode()
    stderr = stderr.decode()
    return stdout


def fetch_pairs(string):
    # The regular expression pattern to match pairs of numbers in parentheses
    pattern = r'\((\d+),(\d+)\)'
    # Find all matches in the string
    matches = re.findall(pattern, string)
    # Convert pairs of strings to pairs of integers
    pairs = [(int(x), int(y)) for x, y in matches]
    return pairs


def replace_pairs(string, replacement):
    # The regular expression pattern to match pairs of numbers in parentheses
    pattern = r'\(\d+,\d+\)'
    # Replace all matches in the string with the replacement string
    new_string = re.sub(pattern, replacement, string)
    return new_string

def fetch_line_content(file_path, line_number):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i == line_number:
                return "'" + line.strip() + "'"
    return 'Line number out of range'

def fetch_line_remain_content(file_path, line_number, column_number):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i == line_number:
                if column_number <= len(line):
                    return line[column_number - 1:].strip()
                else:
                    return 'Column number out of range'
    return 'Line number out of range'

def fetch_next_word(file_path, line_number, column_number):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i == line_number:
                if column_number <= len(line):
                    remain_line_content = line[column_number - 1:].strip()
                    word = remain_line_content.split(' ')[0]
                    return "'" + word + "'"
                else:
                    return 'Column number out of range'
    return 'Line number out of range'

def fetch_between_content(file_path, line_number, col):
    pass


def manul_test():
    error = verify('../Test_Cases/SelectionSort.dfy', 'dafny verify ')
    print(process_error_message('../Test_Cases/SelectionSort.dfy', error))


def process_error_message(generated_file_path, error_message):
    pairs = fetch_pairs(error_message)
    originl_lines = error_message.split('\n')
    lines = []
    final_lines = ""
    for line in originl_lines:
        if '.dfy(' in line:
            line = line.split('): ')[1].strip()
            lines.append(line)
        else:
            lines.append(line)
    for i in range(len(pairs)):
        row, col = pairs[i]
        error_line = lines[i]
        # if 'return path' in error_line:
        #     code_content = fetch_between_content(generated_file_path, row, col)
        if 'postcondition' in error_line or 'postcondition' in error_line:
            code_content = fetch_line_content(generated_file_path, row)
        elif 'Related location' in error_line:
            if len(error_line) <= 18:
                code_content = fetch_next_word(generated_file_path, row, col)
            else:
                code_content = fetch_line_content(generated_file_path, row)
        elif 'index out of range' in error_line:
            code_content = fetch_next_word(generated_file_path, row, col)
        else:
            code_content = fetch_line_content(generated_file_path, row)
        final_line = code_content + ": " + error_line + "\n"
        final_lines += final_line
    final_lines += "\n" + lines[-2]
    if 'unresolved identifier' in final_lines:
        final_lines += '\nHint: remember to include original helper functions or methods (do not change anything) if provided.'
    return final_lines
    # for pair in pairs:
    #     row, col = pair
    #     line_content = fetch_line_content(generated_file_path, row)
    #     to_replace = f'({row},{col})'
    #     error_message = error_message.replace(to_replace, line_content)
    # return error_message


# Helper function to create error examples for dataset
def create_error_examples():
    seq = 1
    max_seq = 5
    system_message = f"""
Add missing loop invariant for this method without change or remove current code. Since your response\
will be tested with verifier, pls be careful and accurate. 

The response should Directly start with valid dafny code and code only. Do not apologize or explain \
when making mistakes. Remeber to put full complete method into a single code block and return the \
code block as your response (if other helper functions provided, include it in your response also). \
Read error message carefully and modify your code accordingly.
    """
    valid_string = 'verified, 0 errors'
    directory = '../dataset/normal_data/prompt/'

    files_without_invariant = []
    test_result = ""
    all_files = os.listdir(directory)
    for file in all_files:
        if file.endswith('.dfy') and file.startswith('MaxPerdV1'):
            files_without_invariant.append(file)
    for file in files_without_invariant:
        ori_method = read_file(directory + file)
        messages=[
          {"role": "system", "content": system_message},
          {"role": "user", "content": ori_method}
        ]
        while seq <= max_seq:
            completion = turbo_completion(messages)
            generated_file_path = f'../Generated_Code/{file}_{seq}.dfy'
            save_file(generated_file_path, completion)
            test_result = verify(generated_file_path, 'dafny verify ')

            if valid_string not in test_result:
                optimized_error_message = process_error_message(generated_file_path, test_result)
                print(f'\033[91mFailed in seq: {seq}\033[0m\n{file}\n{optimized_error_message}\n\n')
                generated_file_path = f'../Generated_Code/{file}_{seq}.dfy'
                save_file(generated_file_path, completion)

                error_file_path = '../dataset/error_data/real_error/' + file.replace('.dfy', '_error_' + str(seq) + '.dfy')
                save_file(error_file_path, completion + '\n\nError:\n' + optimized_error_message)
                
                messages.append({"role": "assistant", "content": completion})
                messages.append({"role": "user", "content": 'But verifier gave error: ' + optimized_error_message})
                seq += 1
            else:
                break
            time.sleep(2)
        if seq > max_seq:
            print(f'\033[92mExceed max seq: {seq}\033[0m\n{file}\n')
        else:
            print(f'\033[92mSucceed in seq: {seq}\033[0m\n{test_result}')
            success_file_path = '../dataset/error_data/real_error/' + file.replace('.dfy', '_success_' + str(seq) + '.dfy')
            save_file(success_file_path, completion)
        seq = 1


def main(file_name):
    test_file_path = '../Test_Cases/' + file_name + '.dfy'
    method = read_file(test_file_path)
    system_message = f"""
Add missing loop invariant for this method without change or remove current code. Since your response\
will be tested with verifier, pls be careful and accurate. 

The response should Directly start with valid dafny code and code only. Do not apologize or explain \
when making mistakes. Remeber to put full complete method into a single code block and return the \
code block as your mesponse. Read error message carefully and modify your code accordingly.
    """
    messages=[
          {"role": "system", "content": system_message},
          {"role": "user", "content": method}
        ]

    completion = turbo_completion(messages)
    loop_seq = 1
    generated_file_path = f'../Generated_Code/{file_name}_{loop_seq}.dfy'
    save_file(generated_file_path, completion)
    test_result = verify(generated_file_path, 'dafny verify ')
    optimized_error_message = process_error_message(generated_file_path, test_result)
    valid_string = 'verified, 0 errors'

    while valid_string not in test_result:
        print(f'\033[91mFailed in seq: {loop_seq}\033[0m\n{optimized_error_message}\n\n')
        generated_file_path = f'../Generated_Code/{file_name}_{loop_seq}.dfy'
        save_file(generated_file_path, completion)
        messages.append({"role": "assistant", "content": completion})
        messages.append({"role": "user", "content": 'But verifier gave error: ' + optimized_error_message})
        save_json(f'./chat_log/{file_name}_{loop_seq}.json', messages)
        loop_seq += 1
        time.sleep(2)
        completion = turbo_completion(messages)
        test_result = verify(generated_file_path, 'dafny verify ')
        optimized_error_message = process_error_message(generated_file_path, test_result)
    print(f'\033[92mSucceed in seq: {loop_seq}\033[0m\n{test_result}')