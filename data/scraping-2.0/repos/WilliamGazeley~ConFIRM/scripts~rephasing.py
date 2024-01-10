import pandas as pd
import os
import argparse
from langchain.chat_models import ChatVertexAI, ChatOpenAI
import re
import glob
import pandas as pd
from langchain.llms import OpenAI
from lora_confirm import rephrase, filters
import warnings


# Instantiate the parser
parser = argparse.ArgumentParser(description=\
    "rephasing script" \
    )
# Required arguments

parser.add_argument('--question_path', type=str, required=True,
                    help='File path to the questions to be rephrased. csv file is required. e.g.: ./datsets/ConFIRM_QAset.csv')
parser.add_argument('--save_path', type=str, required=True,
                    help='Save path (directory) of the rephased question generated. i.e.: ./datsets/')
parser.add_argument('--rephase_llm', type=str, required=True,
                    help='LLM used to rephase the question. i.e.: "gpt-3.5-turbo-instruct", "chat-openai".')
parser.add_argument('--openai_api_key', type=str, required=False,
                    default='None',
                    help='OpenAI API KEY. It is required if you use the model provided by OpenAI.')
parser.add_argument('--google_credential_path', type=str, required=False,
                    default="gcp-service-account.json",
                    help='Path to the gcp-service-account.json. It is required if you use the palm model')
parser.add_argument('--n', type=int, required=False,
                    default=6, help='Number of rephrases to be generated for each question.')


def main(**kwargs):
    """Rephase the questions."""     
    if kwargs['rephase_llm'] == "gpt-3.5-turbo-instruct":
        assert kwargs['openai_api_key'] != 'None', "OpenAI API KEY is required if you use the model provided by OpenAI."
        os.environ['OPENAI_API_KEY'] = kwargs['openai_api_key']
        llm = OpenAI(model='gpt-3.5-turbo-instruct') 
    elif kwargs['rephase_llm'] == "chat-openai":
        assert kwargs['openai_api_key'] != 'None', "OpenAI API KEY is required if you use the model provided by OpenAI."
        os.environ['OPENAI_API_KEY'] = kwargs['openai_api_key']
        warnings.warn("Use 'gpt-3.5-turbo-instruct' as rephase llm to get better rephase performance.", UserWarning)
        llm = ChatOpenAI()
    elif kwargs['rephase_llm'] == "palm":
        import google.auth
        assert os.path.exists(kwargs['google_credential_path']), f"Invalid google credential path {kwargs['google_credential_path']}"
        credentials, project_id = google.auth.load_credentials_from_file(kwargs['google_credential_path'])
        llm = ChatVertexAI()
    else:
        raise NotImplementedError("Only gpt-3.5-turbo-instruct and chat-openai are available now.")
     
    assert os.path.exists(kwargs['question_path']), f"Invalid question data path {kwargs['question_path']}"  
    df = pd.read_csv(kwargs['question_path'])
    
    if 'quality' in df.columns:
        df = df[df['quality'] == 2]
    else:
        warnings.warn("Quality column is missing. All questions will be rephrased.", UserWarning)

    # If previous batches exist, start from there instead of from scratch
    files = glob.glob('ocean_rephrased_*.csv')
    output_dir = kwargs['save_path']
    highest_batch_num = -1

    # Find the highest batch number from the existing CSV files
    for file in files:
        match = re.search(fr'{output_dir}/ocean_rephrased_(\d+)_(\d+)\.csv', file)
        if match:
            batch_num = int(match.group(1))
            if batch_num > highest_batch_num:
                highest_batch_num = batch_num

    # Start the outer loop from the highest batch number + 1
    start_batch = highest_batch_num + 1

    for i in range(start_batch, kwargs["n"]):
        # Break the dataset into batches of 50 questions (checkpointing)
        for j in range(0, len(df), 50):
            batch = df.iloc[j:j+50]
            batch = rephrase.ocean(llm=llm, df=batch, 
                                sample_method="specific", n=10)
            batch.to_csv(f"{output_dir}/ocean_rephrased_{i}_{j}.csv", index=False)

    # Concat all the batches
    all_files = glob.glob(f"{output_dir}/ocean_rephrased_*.csv")
    all_dfs = []
    for filename in all_files:
        all_dfs.append(pd.read_csv(filename))
    df = pd.concat(all_dfs)

    # Filters
    df = df[df['rephrase'].notna()]
    df = filters.rouge_l(df=df, columns=['rephrase'], threshold=0.7)
    df = filters.rouge_l(df=df, columns=['rephrase', 'expected_fields_with_descriptions'], axis=1, threshold=0.3)
    df['question'] = df['rephrase']
    df.drop(columns=['rephrase', 'quality', 'prompts', 'expected_fields_with_descriptions'], inplace=True, errors='ignore')
    
    # Save file
    q_filename = kwargs['question_path'].split('/')[-1]
    df.to_csv(f"{output_dir}/{q_filename.replace('.csv', '_ocean_rephrased.csv')}", index=False)

    # Clean up the batch files
    for filename in all_files:
        if os.path.exists(filename): os.remove(filename)


if __name__=="__main__":
    args = parser.parse_args()
    main(**vars(args))