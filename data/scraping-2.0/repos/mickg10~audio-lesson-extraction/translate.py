
import translateopenai
import argparse
import logging
import pandas as pd
import sys
from openai import OpenAI

def main() -> int:
    parser = argparse.ArgumentParser()
    try:
        client = OpenAI(api_key=translateopenai.read_openai_key("../openai.key"))
    except:
        pass
    parser.add_argument("--log_level", default=logging.INFO, type=lambda x: getattr(logging, x), help=f"Configure the logging level: {list(logging._nameToLevel.keys())}")
    parser.add_argument('--input', help=f"what file to use")
    parser.add_argument('--output', help=f"where to write the output files")
    parser.add_argument('--language', default='english', help="What language to use")
    parser.add_argument('--context', type=int,default=40,help="How many lines to process")
    parser.add_argument('--model', default="gpt-3.5-turbo",help=f"What openai model to use - {[m for m in [x['id'] for x in client.models.list()['data']] if 'gpt' in m]}")
    parser.add_argument('--api_key_file', default="../aikeys/openai.key",help="OPENAI_API_KEY file")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format='%(asctime)s:%(lineno)d %(message)s')
    
    df = pd.read_excel(args.input, engine='openpyxl')
    def write_df(x):
        if (args.output.endswith(".xlsx")):
            x.to_excel(args.output, index=False)
        elif (args.output.endswith(".txt")): 
            x.to_csv(args.output, index=False, sep="\t")
    translateopenai.translate_dataframe(args.api_key_file, df, speaker_col="speaker", text_col="text", output_col="translated", language=args.language, context_lines=args.context, model=args.model, apply_at_step=write_df)
    write_df(df)
    return 0

if __name__ == '__main__':
    sys.exit(main())

