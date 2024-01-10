import argparse
import openai

def get_args_parser():
    parser = argparse.ArgumentParser('Finetune via OpenAI API')
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--train_filename", type=str, help="Training jsonl dataset")
    return parser

def upload(train_filename):    
    response = openai.File.create(
    file=open(train_filename, "rb"),
    purpose='fine-tune'
  ) 
    print(response)

def main(args):
    openai.api_key = args.api_key
    upload(args.train_filename)

if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)