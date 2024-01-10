import argparse
import json
import openai

def main(args):
    with open(args.sweep_path, 'r') as f: # jsonl
        sweep = [json.loads(line) for line in f.readlines()]
    
    for i,run in enumerate(sweep):
        print('Run', i)
        print(openai.FineTuningJob.retrieve(run['run_id']))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--sweep_path', help='Description for argument 1')
    args = parser.parse_args()
    main(args)