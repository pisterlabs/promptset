import time
import os
import openai
import argparse
from random import choice
from utils import Logger
from ROSPackageSearchPromptLoader import ROSPackageSearchPromptLoader

class ChatGPT:
    def __init__(self, prompt_loader, logger, repeat=5):
        self.prompt_loader = prompt_loader
        self.logger = logger
        self.repeat = repeat

    def send_request(self, message):
        keys = [
            "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",
            "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",
            "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",
            "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",
        ]
        openai.api_key = choice(keys)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                timeout=120,
            )
        except Exception as e:
            response = {"Exception:": e}
        return response

    def start_conversation(self):
        results = []
        for idx, message in self.prompt_loader.generate():
            _results = []
            if idx % 5 == 0 and idx != 0:
                time.sleep(1)
            else:
                time.sleep(1)
            for i in range(self.repeat):
                response = self.send_request(message)
                # time.sleep(20)
                try:
                    response_data = response["choices"][0]["message"]["content"]
                    self.logger.info(f"Total tokens in response: {response['usage']['total_tokens']}")
                    _results.append(response_data)
                except Exception:
                    self.logger.error(response)
            results.append((idx, _results))
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_dir', type=str, help='path to code file')
    parser.add_argument('--log_dir', type=str, help='path to log file')
    parser.add_argument('--output_dir', type=str, help='path to output file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, help='end index')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613', help='model name')
    args = parser.parse_args()

    if args.start < 0:
        raise ValueError('start index must be non-negative')
    if args.end < args.start:
        raise ValueError('end index must be greater than start index')
    if not os.path.exists(args.code_dir):
        print(args.code_dir)
        raise ValueError('code file does not exist')

    logger = Logger(args.log_dir, 'debug').logger


    loop_num = (args.end - args.start - 1) // 10 + 1
    for i in range(loop_num):
        start = args.start + i * 10
        if i == loop_num - 1:
            end = args.end
        else:
            end = args.start + (i + 1) * 10
        print(f'Processing data from {start} to {end}.')

        t0 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if args.model:
            loader = ROSPackageSearchPromptLoader(args.code_dir, logger, start, end, args.model)
        else:
            loader = ROSPackageSearchPromptLoader(args.code_dir, logger, start, end)

        chatgpt = ChatGPT(loader, logger, 1)

        t1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        results = chatgpt.start_conversation()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for idx, res in results:
            for j in range(len(res)):
                with open(os.path.join(args.output_dir, f'package_search_{idx}_{j}.txt'), 'w') as f:
                    f.write(res[j])
        t2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        print('start time: ', t0)
        print('conversation start: ', t1)
        print('end time: ', t2)

# python ./RunChatGPT.py --code_dir ./test_region_word_in_examples.csv --log_dir test_phrase.log --output_dir test_node_output --start 0 --end 24\n
# python ./RunChatGPT_evaluation_search.py --code_dir ../pkg_recommendation/user_queries.csv --log_dir test_package_search_0_10.log --output_dir search_result --start 0 --end 1