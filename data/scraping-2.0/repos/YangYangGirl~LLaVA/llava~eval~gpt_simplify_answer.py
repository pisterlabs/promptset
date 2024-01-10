import argparse
import json
import os

import openai
import tqdm
import ray
import time

NUM_SECONDS_TO_SLEEP = 3

@ray.remote(num_cpus=4)
def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for simplify the given caption.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    print('success!')
    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-i', '--input-file')
    parser.add_argument('-o', '--output-file')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    ray.init()
    max_tokens = args.max_tokens
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output_file = open(f'{args.output_file}', 'w')
    
    raw_captions = [json.loads(q) for q in open(os.path.expanduser(args.input_file), "r")]
    # import pdb; pdb.set_trace()
    for idx in range(0, len(raw_captions), 3):
        print(idx)
        content = raw_captions[idx]["text"]

        times = 0
        max_times = 5
        while times <= max_times:
            times += 1
            try:
                response = openai.ChatCompletion.create(
                        model='gpt-4',
                        messages=[{
                            'role': 'system',
                            'content': 'You are a helpful and precise assistant to make the given caption concise.'
                        }, {
                            'role': 'user',
                            'content': content,
                        }],
                        temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                        max_tokens=max_tokens)

                # To avoid the rate limit set by OpenAI
                raw_captions[idx]["simplify_text"] = response["choices"][0]["message"]["content"]
            except:
                time.sleep(30)
                print ("retry, times: %s/%s" % (times, max_times))

        output_file.write(json.dumps(raw_captions[idx]) + '\n')
        
        # import pdb; pdb.set_trace()
        # time.sleep(NUM_SECONDS_TO_SLEEP)
    output_file.close()
    # for idx, review in enumerate(reviews):
    #     scores = parse_score(review)
    #     js_list[idx]['content'] = review
    #     js_list[idx]['tuple'] = scores
    #     review_file.write(json.dumps(js_list[idx]) + '\n')
    # review_file.close()
