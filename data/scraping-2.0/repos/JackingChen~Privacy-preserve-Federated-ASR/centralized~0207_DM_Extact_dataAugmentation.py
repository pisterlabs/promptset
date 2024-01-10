import os
import argparse
import pandas as pd
from addict import Dict

from rouge_score import rouge_scorer
import pandas as pd
import argparse 
import os
from bleu import list_bleu
from fastchat.model import load_model, get_conversation_template
from langchain.prompts import PromptTemplate
import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
from addict import Dict

import openai
#importing the necessary dependencies
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, FewShotPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain import hub
from rouge_score import rouge_scorer
import subprocess

from langchain.embeddings import AzureOpenAIEmbeddings
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('-hyperparemeter','--Aug_k', type=int, default=1, help="")
parser.add_argument('--summary_dir_in', type=str, default="/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-anomia", help="")
parser.add_argument('--Outpath_root', type=str, default="/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Augment_data/", help="")
args = parser.parse_args()
top_k=args.Aug_k

df_train = pd.read_pickle(f"{args.summary_dir_in}/train.pkl")
df_val = pd.read_pickle(f"{args.summary_dir_in}/dev.pkl")
df_test = pd.read_pickle(f"{args.summary_dir_in}/test.pkl")


#########====================end Retreiver area==============================
class RAG_chatbot:
    def __init__(self):
        self.retreiver = None
        self.stepback_model = None
        self.answer_model = None
        self.few_shot_prompt=None

    def Initialize_openai(self,env_script_path = './env.sh'):
        #activate environment
        # 读取 env.sh 文件内容
        with open(env_script_path, 'r') as file:
            env_content = file.read()

        # 将 env.sh 文件内容以换行符分割，并逐行执行
        for line in env_content.split('\n'):
            # 跳过注释和空行
            if line.strip() and not line.startswith('#'):
                # 使用 split 等号来分割键值对
                key, value = line.split('=', 1)
                # 设置环境变量
                os.environ[key.replace("export ","")] = value.strip()
        #Initialize openai
        openai.api_type = "azure"
        openai.api_version = "2023-05-15" 
        openai.api_base = os.getenv('OPENAI_API_BASE')  # Your Azure OpenAI resource's endpoint value.
        openai.api_key = os.getenv('OPENAI_API_KEY')
        chatopenai=ChatOpenAI(api_key=openai.api_key,model_kwargs={"engine": "gpt-35-turbo"})
        return chatopenai
    def Initialize_Embedder(self):
        os.environ["AZURE_OPENAI_API_KEY"] = openai.api_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = openai.api_base


        embedder = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2023-05-15",
        )
        return embedder

    def Initialize_fewshot_prompt(self, user_input):
        # 在知識庫中搜尋與使用者輸入相關的資訊
        # 這裡假設 knowledge_base 是一個包含資訊的字典或其他數據結構
        if user_input in self.knowledge_base:
            return self.knowledge_base[user_input]
        else:
            return None
RAG_bot=RAG_chatbot()
chatopenai=RAG_bot.Initialize_openai()
Embedder=RAG_bot.Initialize_Embedder()


from prompts import assesmentPrompt_template, Instruction_templates, Psychology_template,\
    Sensitive_replace_dict, generate_psychology_prompt


def process_sessions(session_df, prompt_template, chatopenai, Sensitive_replace_dict, Embedder, use_text='text'):
    Embedding_dict, Summary_dict, Prompt_dict = {}, {}, {}
    for session, row in session_df.iterrows():
        if session in Sensitive_replace_dict.keys():
            dialogue_content = row[use_text]
            for values in Sensitive_replace_dict[session]:
                dialogue_content = dialogue_content.replace(values[0], values[1])
            
        else:
            dialogue_content = row[use_text]

        prompt=prompt_template.format(dialogue_content=dialogue_content)
        ans_middle = chatopenai.invoke(prompt)

        output_parser = StrOutputParser()
        summary = output_parser.parse(ans_middle).content
        embeddings = Embedder.embed_query(summary)
        Embedding_dict[session] = embeddings
        Summary_dict[session] = summary
        Prompt_dict[session] = prompt

    session_df['Embedding'] = session_df.index.to_series().apply(lambda x: Embedding_dict.get(x, []))
    session_df['Psych_Summary'] = session_df.index.to_series().apply(lambda x: Summary_dict.get(x, []))
    session_df['Psych_Prompt'] = session_df.index.to_series().apply(lambda x: Prompt_dict.get(x, []))
    return session_df

# replicate_cols=['path', 'text', 'dementia_labels', 'pred_str', 'ID   ']
prompt_template=assesmentPrompt_template['data_augmentation'].format(**{'content': '{dialogue_content}'})



def augment_sessions(session_df, args, Sensitive_replace_dict, prompt_template, chatopenai,\
                      use_text='text',\
                      replicate_cols=['path', 'text', 'dementia_labels', 'pred_str', 'ID   ']
                    ):
    session_aug_df = pd.DataFrame(columns=replicate_cols)
    failing_session = []

    for session, row in session_df.iterrows():
        for i in range(args.Aug_k):
            suffix = f"_aug{i}"
            new_session_name = f"{session}{suffix}"
            session_aug_df.loc[new_session_name] = row

            if session in Sensitive_replace_dict.keys():
                dialogue_content = row[use_text]
                for values in Sensitive_replace_dict[session]:
                    dialogue_content = dialogue_content.replace(values[0], values[1])
            else:
                dialogue_content = row[use_text]

            prompt = prompt_template.format(dialogue_content=dialogue_content)

            try:
                ans_middle = chatopenai.invoke(prompt)
                output_parser = StrOutputParser()
                summary = output_parser.parse(ans_middle).content
                session_aug_df.loc[new_session_name, 'Aug_Prompt'] = prompt
                session_aug_df.loc[new_session_name, 'text'] = summary
            except:
                failing_session.append({session: dialogue_content})
                continue

    return session_aug_df, failing_session
session_aug_df, failing_sessions = augment_sessions(df_train, args, Sensitive_replace_dict, prompt_template, chatopenai)


print("Failing number" ,len(failing_sessions))
print(failing_sessions)

base_dirname=os.path.basename(args.summary_dir_in)
os.makedirs(f"{args.Outpath_root}/{base_dirname}",exist_ok=True)
Outpath=f"{args.Outpath_root}/{base_dirname}/train.pkl"
session_aug_df.to_pickle(Outpath)