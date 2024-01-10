import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from threading import Thread

import datasets

import openai
from code_synthesis_exercises import synthesize_exercises
from openai_api_wrapper import OpenaiAPIWrapper

with open('words.txt','r') as f:
    words_set = f.readlines()
    
words_set = [word.strip() for word in words_set]

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--api_key",type=str, required=True, help="Your OpenAI API key")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for generated output")
    parser.add_argument('--gen_nums',type=int, default=2000)
    parser.add_argument('--output_dir',type=str,default='./exercise')
    parser.add_argument("--threads_num_per_key", type=int, default=160)
    return parser.parse_args()

def save_response_to_file(worker_id, response, output_dir):
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d%H%M%S')
    filename = f'response_{timestamp}_worker{worker_id}.json'

    dialogue_data = {
        "response": response
    }

    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dialogue_data, f, ensure_ascii=False, indent=4)
        
def generate_textbooks(worker_id, args,api_keys):
    llm = OpenaiAPIWrapper()
    current_key = api_keys[int(worker_id%4)]
    llm.set_api_key(current_key) # 200个线程，每50个线程用一个key
    start_time = time.time()
    system_prompt = '''You are a helpful assistant. '''
    user_prompt = synthesize_exercises(words_set,num_words = 5)
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_prompt})
    generated_num = 0
    while generated_num < args.gen_nums:
        try:
            output = llm.call_turbo_using_messages(messages, max_tokens=args.max_tokens, temperature=1.0, top_p=0.9)
            response = llm.parse_chatgpt_response(output)
            generated_num += 1
        except Exception as e:
            print('Unexpected error')
            print('Exception: {}'.format(e))
            if 'exceeded your current quota' in str(e) or 'due to violation of our policies' in str(e):
                with open('./failed_api_keys.txt', 'a') as f:
                    f.write('{}\n'.format(current_key))
                # 读取文件内容
                with open('./failed_api_keys.txt', 'r') as file:
                    lines = file.readlines()
                # 去除重复项
                unique_lines = list(set(lines))
                # 重新储存到新文件
                with open('failed_api_keys.txt', 'w') as file:
                    file.writelines(unique_lines)
                # 当前环境下的api列表中去除当前失效key
                api_keys.remove(current_key)
                if len(api_keys) >= 4:
                    llm.set_api_key(api_keys[int(worker_id%4)])
                else:
                    print('Available apis < 4.')
                    return None
            continue
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        save_response_to_file(worker_id,response,args.output_dir)
        print('# Worker_{} collected schemas: {}'.format(
                worker_id, generated_num))
        

def main(args):
    start_time = time.time()
    threads = []
    with open('extracted_keys.txt','r') as f:
        api_keys = [key.strip() for key in f.readlines()]
    for j in range(args.threads_num_per_key):
        t = Thread(target=generate_textbooks, args=(j, args,api_keys))
        t.start()
        print(str(j)+"starts!")
        threads.append(t)
    for t in threads:
        t.join()
    total_time = time.time() - start_time
    print(f'Generate Finshed! with {total_time}')

if __name__ == '__main__':
    args = parse_args()
    main(args)