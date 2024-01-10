import argparse
import os
from tqdm import tqdm
import json
import openai

def main(args):
    openai.api_key = args.api_key

    ## load data from input file.
    with open(args.input_path) as f:
        data = json.load(f)
    
    ## load jsfusion val split vids from split file.
    with open(args.split_path) as f:
        lines = f.readlines()[1:]
    jsfusion_vids = set([l_i.split(',')[2] for l_i in lines])
    
    ann = {}
    for sample in data['sentences']:
        vid = sample['video_id']
        if vid not in jsfusion_vids:
            ## skip if the sample is not included in jsfusion val split.
            continue
        if vid not in ann:
            ann[vid] = []
        ann[vid].append(sample['caption'])
    print(f"# original samples: {len(ann)}")

    out = {}

    if os.path.isfile(args.out_path):
        with open(args.out_path) as f:
            out = json.load(f)

    check_set = set(out.keys())

    for i, (key, val) in enumerate(tqdm(ann.items())):
        if key in check_set:
            continue
        messages = [
            {
                "role": "system",
                "content": "After watching the same 15-second video, twenty annotators each wrote captions. Based on their captions, infer the content of the video and create a single caption that comprehensively encapsulates the video's content without arbitrarily adding information that is not present in the annotators' captions. However, ensure that all the information mentioned in the captions provided by the annotators is included in the final caption."
            },
            {
                "role": "user",
                "content": "\n".join(val)
            }
        ]
        for _ in range(10):
            try:
                res = openai.ChatCompletion.create(
                            model='gpt-3.5-turbo',
                            messages=messages,
                            temperature=0.2,
                            max_tokens=1024
                )
                unified = res['choices'][0]['message']['content']

                out[key] = unified
                with open(args.out_path, 'w') as f:
                    f.write(json.dumps(out, indent=4))
                break
            except Exception as e:
                print(e)
                continue
    print(f"# processed samples: {len(out)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="make MSR-VTT gt json file.")
    parser.add_argument("--input_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--split_path", required=True, help="The path to file containing split info.")
    parser.add_argument("--out_path", required=True, help="The path to save gt json files.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    args = parser.parse_args()
    main(args)