import pandas as pd
import sys
import os
import openai
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()
pd.options.display.max_columns = None
summaries_clean_700_to_710_with_gpt = 'summaries_clean_700_to_710_with_gpt.pkl'

openai.api_key = os.getenv("OPENAI_API_KEY")
PROMPT = 'Summarize and expand the following story to make it original, surreal, and magical:\n\n'

def get_gpt_summary(summary_clean):
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=f"{PROMPT}{summary_clean}",
        temperature=1,
        max_tokens=300,
        top_p=0.5,
        frequency_penalty=1,
        presence_penalty=1
    )
    return response.get('choices')[0]['text']

if __name__ == "__main__":
    if Path(summaries_clean_700_to_710_with_gpt).is_file():
        print(f"aborting because processed file {summaries_clean_700_to_710_with_gpt} already exists")
        sys.exit(0)
        
    df = pd.read_pickle('summaries_clean.pkl', compression='xz')

    # 225 stories
    df = df[(df.summary_length > 700) & (df.summary_length < 710)]
    df.reset_index(drop=True, inplace=True)

    # call GPT-3 and generate new summary for each plot
    df['gpt_summary'] = df.summary_clean.progress_apply(get_gpt_summary)

    df.to_pickle(summaries_clean_700_to_710_with_gpt)
