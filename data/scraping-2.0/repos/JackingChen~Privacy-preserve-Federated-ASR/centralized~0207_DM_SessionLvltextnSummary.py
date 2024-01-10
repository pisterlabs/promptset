import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from collections import Counter
import pickle
import random
import argparse
import time
from datetime import datetime

# torch:
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers import BertTokenizer, BertConfig, BertModel,XLMTokenizer, XLMModel
from prompts import assesmentPrompt_template, Instruction_templates, Psychology_template,\
    Sensitive_replace_dict, generate_psychology_prompt
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, FewShotPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from Dementia_challenge_models import Embsize_map
class Arg:
    version = 1
    # data
    epochs: int = 5  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 350  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    a_hidden_size = 768 # BERT-base: 768, BERT-large: 1024, BERT paper setting
    t_hidden_size = 768
    t_x_hidden_size = a_hidden_size+t_hidden_size
    batch_size: int = 8
            
class BertPooler(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

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
        from langchain.embeddings import AzureOpenAIEmbeddings
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




class Model(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        # config:
        
        self.args = args
        self.config = config
        self.batch_size = self.args.batch_size
        
        # meta data:
        self.epochs_index = 0
        self.label_cols = 'dementia_labels'
        self.label_names = ['Control','ProbableAD']
        self.num_labels = 2
        self.t_embed_type = self.config['t_embed']
        self.a_embed_type = self.config['a_embed']
        self.a_hidden = self.args.a_hidden_size

        if self.config['process_summary']:
            self.RAG_bot=RAG_chatbot()
            self.chatopenai=self.RAG_bot.Initialize_openai()
            prompts_dict = generate_psychology_prompt(assessment_prompt_template=assesmentPrompt_template,
                                            instruction_templates=Instruction_templates,
                                            psychology_template=Psychology_template,
                                            )
            self.result_prompts=prompts_dict['self.config.selected_psych']


        
        # --> HERE STEP 1 <--
        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.test_step_outputs = []   # save outputs in each batch to compute metric overall epoch
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch


        if self.t_embed_type == "mbert":
            self.t_hidden = self.args.t_hidden_size
            
            t_pretrained = 'bert-base-multilingual-uncased'
            self.t_tokenizer = BertTokenizer.from_pretrained(t_pretrained)
            self.t_model = BertModel.from_pretrained(t_pretrained)
            
            
        elif self.t_embed_type == "xlm":
            self.t_hidden = self.args.t_hidden_size
            
            t_pretrained = 'xlm-mlm-100-1280'
            self.t_tokenizer = XLMTokenizer.from_pretrained(t_pretrained)
            self.t_model = XLMModel.from_pretrained(t_pretrained)
            self.pooler = BertPooler(self.t_hidden)
            
        self.hidden = int(self.t_hidden + self.t_hidden)
        
        if self.a_embed_type == "en":
            a_pretrained =  "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            
        elif self.a_embed_type == "gr":
            a_pretrained =  "lighteternal/wav2vec2-large-xlsr-53-greek"

        elif self.a_embed_type == "multi":
            a_pretrained = "voidful/wav2vec2-xlsr-multilingual-56"
            
        elif self.a_embed_type == "wv":
            a_pretrained ='facebook/wav2vec2-base'
            
        self.a_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(a_pretrained)
        self.a_model = Wav2Vec2Model.from_pretrained(a_pretrained)
        
        
        self.clf1 = nn.Linear(self.hidden, int(self.hidden/2))
        self.clf2 = nn.Linear(int(self.hidden/2), self.num_labels)
        
            
            
    # def forward(self, text, audio):
    def forward(self, text1, text2):
        
        if self.t_embed_type == "mbert":
            t1_out = self.t_model(text1)[1] 

            t2_out = self.t_model(text2)[1] 

            
        elif self.t_embed_type == "xlm":
            t1_out = self.t_model(text1)[0]
            t1_out = self.pooler(t1_out)

            t2_out = self.t_model(text2)[0]
            t2_out = self.pooler(t2_out)
            
            
        # a_out = self.a_model(audio)['extract_features']#[2] #last_hidden_state , feature extraction
        # a_out = a_out[:, 0, :] 
        
        #print(a_out)
        #print(a_out['extract_features'].shape) # ([8, 437, 512])
        #print(a_out['last_hidden_state'].shape) # ([8, 437, 1024]) => pooling 필요
        
        
        output = torch.cat((t1_out,t2_out),axis=1)   
        # output = t_out
        #print(output.shape)
        
        logits = self.clf2(self.clf1(output))
    
        return logits
        

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def preprocess_dataframe(self):
        tg_sr = 16000
        t_col_name = "text" 
        # a_col_name = "path"         
        # df = pd.read_json('/mnt/Internal/FedASR/Data/230126_total_asr_data.json')
        df = pd.read_csv(config.file_in)

        def Augment_info_df(df_test):
            # 將 'path' 欄位進行字串操作
            df_test['path'] = df_test['path'].str.rstrip('.wav')

            # 使用 str.split 拆分 'path' 欄位
            df_test[['session', 'role', 'number', 'start_time', 'end_time']] = df_test['path'].str.split('_', expand=True)

            # 如果 'number' 欄位的末尾包含 '.wav'，進行一次額外的拆分
            df_test['number'] = df_test['number'].str.rstrip('.wav')

            # 將 'start_time' 和 'end_time' 欄位轉換為數值型別
            df_test[['start_time', 'end_time']] = df_test[['start_time', 'end_time']].astype(int)

            return df_test
        df = Augment_info_df(df)


        # Packer
        def Packer(df_test) -> dict:
            People_dict = dict(tuple(df_test.groupby('session')))
            return People_dict

        def Dialogueturn2corpus(data_frame, mode='text'): #mode can be 'pred_str' or 'text'
            # 按 'start_time' 列進行排序
            sorted_data_frame = data_frame.sort_values(by='start_time')

            # 添加前綴並使用 '\n' 進行拼接
            processed_text = sorted_data_frame.apply(lambda row: f"{row['role']}: {row[mode]}", axis=1).str.cat(sep='\n')

            return processed_text

        def filter_people_dict(People_dict, mode="INV+PAR", verbose=False) -> dict:# 'INV' , 'PAR', 'INV+PAR'
            filtered_people_dict = {}

            for session, data_frame in People_dict.items():
                # 使用 query 過濾 'role' 為 'INV' 或 'PAR'
                if mode == "PAR":
                    filtered_data_frame = data_frame.query("role == 'PAR'")
                    filtered_people_dict[session] = filtered_data_frame
                elif mode == "INV":
                    filtered_data_frame = data_frame.query("role == 'INV'")
                    filtered_people_dict[session] = filtered_data_frame
                elif mode == "INV+PAR":
                    filtered_people_dict[session] = data_frame
                else:
                    raise OSError
            
            if verbose:
                # 印出過濾後的 People_dict 中每個 session 的 DataFrame
                for session, data_frame in filtered_people_dict.items():
                    print(f"Session: {session}")
                    print(data_frame)
                    print("\n")

            return filtered_people_dict

        # Dialogue Formatter
        def Dialogue_Formatter(People_dict, sep="\n",role_mode='PAR')->dict:
            session_df=pd.DataFrame()
            for session, data_frame in People_dict.items():
                if len(data_frame)>0:
                    total_info=data_frame.iloc[0].copy()
                    sessional_text = Dialogueturn2corpus(data_frame,mode='text')
                    sessional_predStr = Dialogueturn2corpus(data_frame,mode='pred_str')
                    
                    # total_info,'text']=sessional_text
                    # session_df.loc[session,'pred_str']=sessional_predStr
                    # session_df.loc[session,'role']=role_mode
                    # session_df.loc[session,'start_time']=data_frame['start_time'].min()
                    # session_df.loc[session,'end_time']=data_frame['end_time'].max()

                    total_info['text']=sessional_text
                    total_info['pred_str']=sessional_predStr
                    total_info['role']=role_mode
                    total_info['start_time']=data_frame['start_time'].min()
                    total_info['end_time']=data_frame['end_time'].max()
                    session_df = pd.concat([session_df, pd.DataFrame([total_info], index=[session])])
                else:
                    print(f"Session {session} has no data")
            return session_df
        df_train = df[df['ex'] == 'train']
        df_val = df[df['ex'] == 'dev']
        df_test = df[df['ex'] == 'test']

        def SentenceLvldf2SessionLvldf(df, role_mode="PAR"):
            People_dict=Packer(df)
            People_dict = filter_people_dict(People_dict, mode=role_mode, verbose=False)
            df_dialogue=Dialogue_Formatter(People_dict,role_mode)
            return df_dialogue

        df_train=SentenceLvldf2SessionLvldf(df_train)
        df_val=SentenceLvldf2SessionLvldf(df_val)
        df_test=SentenceLvldf2SessionLvldf(df_test)

        def Tokenize(df_data):
            df_data[t_col_name] = df_data[t_col_name].map(lambda x: self.t_tokenizer.encode(
                str(x),
                padding = 'max_length',
                max_length=self.args.max_length,
                truncation=True,
                ))
            return df_data
        df_train=Tokenize(df_train)
        df_val=Tokenize(df_val)
        df_test=Tokenize(df_test)
        df_test = df_test.reset_index(drop=True)

        # audio_root="/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/clips"
        # # 원래 길이: 562992, batch 16: 90000, batch 8: 140000
        # # max_length=16000, truncation=True 이건 일단 돌려보고 결정 => 뒤쪽, 앞에쪽 뭐보면 좋을 지 그런거 check하면 좋으니까! 
        # df[a_col_name] = df[a_col_name].map(lambda x: self.a_tokenizer(
        #     f"{audio_root}/{x}",
        #     sampling_rate = tg_sr,
        #     max_length=100000, 
        #     truncation=True
        #     )['input_values'][0])
        def get_sessiondf_summary(session_df, prompt_template, chatopenai, Sensitive_replace_dict, use_text='text'):
            Summary_dict, Prompt_dict = {}, {}, {}
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
                Summary_dict[session] = summary
                Prompt_dict[session] = prompt

            session_df['Psych_Summary'] = session_df.index.to_series().apply(lambda x: Summary_dict.get(x, []))
            session_df['Psych_Prompt'] = session_df.index.to_series().apply(lambda x: Prompt_dict.get(x, []))
            return session_df
        if self.process_summary:
            df_train=get_sessiondf_summary(df_train, self.result_prompts, self.chatopenai, Sensitive_replace_dict, use_text='text')
            df_val=get_sessiondf_summary(df_val, self.result_prompts, self.chatopenai, Sensitive_replace_dict, use_text='text')
            df_test=get_sessiondf_summary(df_test, self.result_prompts, self.chatopenai, Sensitive_replace_dict, use_text='text')

        self.train_data = TensorDataset(
            torch.tensor(df_train[t_col_name].tolist(), dtype=torch.long),
            # torch.tensor(df_train[a_col_name].tolist(), dtype=torch.float),
            torch.tensor(df_train[self.label_cols].tolist(), dtype=torch.long),
        )
        
        self.val_data = TensorDataset(
             torch.tensor(df_val[t_col_name].tolist(), dtype=torch.long),
            #  torch.tensor(df_val[a_col_name].tolist(), dtype=torch.float),
            torch.tensor(df_val[self.label_cols].tolist(), dtype=torch.long),
        )

        self.test_data = TensorDataset(
             torch.tensor(df_test[t_col_name].tolist(), dtype=torch.long),
            #  torch.tensor(df_test[a_col_name].tolist(), dtype=torch.float),
            torch.tensor(df_test[self.label_cols].tolist(), dtype=torch.long),
             torch.tensor(df_test.index.tolist(), dtype=torch.long),
        )
    
    def preprocess_loaded_summaries(self):
        df_train = pd.read_pickle(f"{config.summary_dir_in}/train.pkl")
        df_val = pd.read_pickle(f"{config.summary_dir_in}/dev.pkl")
        df_test = pd.read_pickle(f"{config.summary_dir_in}/test.pkl")

        t1_col_name='text'
        t2_col_name='Psych_Summary'
        def Tokenize(df_data, t_col_name='text'):
            df_data[t_col_name] = df_data[t_col_name].map(lambda x: self.t_tokenizer.encode(
                str(x),
                padding = 'max_length',
                max_length=self.args.max_length,
                truncation=True,
                ))
            return df_data
        df_train=Tokenize(df_train,t_col_name=t1_col_name)
        df_train=Tokenize(df_train,t_col_name=t2_col_name)
        df_val=Tokenize(df_val,t_col_name=t1_col_name)
        df_val=Tokenize(df_val,t_col_name=t2_col_name)
        df_test=Tokenize(df_test,t_col_name=t1_col_name)
        df_test=Tokenize(df_test,t_col_name=t2_col_name)
        df_test = df_test.reset_index(drop=True)

        self.train_data = TensorDataset(
            torch.tensor(df_train[t1_col_name].tolist(), dtype=torch.long),
            torch.tensor(df_train[t2_col_name].tolist(), dtype=torch.long),
            torch.tensor(df_train[self.label_cols].tolist(), dtype=torch.long),
        )

        self.val_data = TensorDataset(
                torch.tensor(df_val[t1_col_name].tolist(), dtype=torch.long),
             torch.tensor(df_val[t2_col_name].tolist(), dtype=torch.long),
            torch.tensor(df_val[self.label_cols].tolist(), dtype=torch.long),
        )

        self.test_data = TensorDataset(
            torch.tensor(df_test[t1_col_name].tolist(), dtype=torch.long),
            torch.tensor(df_test[t2_col_name].tolist(), dtype=torch.long),
            torch.tensor(df_test[self.label_cols].tolist(), dtype=torch.long),
            torch.tensor(df_test.index.tolist(), dtype=torch.long),
        )
    def train_dataloader(self):
        
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        token1, token2, labels = batch  
        # token,  labels = batch  
        logits = self(token1, token2) 
        # logits = self(token) 
        loss = nn.CrossEntropyLoss()(logits, labels)   
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        token1, token2, labels = batch  
        # token,  labels = batch  
        logits = self(token1, token2) 
        loss = nn.CrossEntropyLoss()(logits, labels)     
        
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        # --> HERE STEP 2 <--
        self.val_step_outputs.append({
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        })
        # self.val_step_targets.append(y_true)
        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }
        
            
    def test_step(self, batch, batch_idx):
        token1, token2, labels,id_ = batch  
        # token,  labels = batch  
        logits = self(token1, token2) 
        # logits = self(token) 
        
        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        # --> HERE STEP 2 <--
        self.test_step_outputs.append({
            'y_true': y_true,
            'y_pred': y_pred,
        })
        # self.test_step_targets.append(y_true)
        return {
            'y_true': y_true,
            'y_pred': y_pred,
        }
    
    def on_validation_epoch_end(self):
        loss = torch.tensor(0, dtype=torch.float)
        # print("Value= ",self.val_step_outputs)
        # print("type(self.val_step_outputs)=",type(self.val_step_outputs))
        # print("type(self.val_step_outputs[0])=",type(self.val_step_outputs[0]))
        # print("type(self.val_step_outputs[0] loss)=",type(self.val_step_outputs[0]['loss']))
        for i in self.val_step_outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(self.val_step_outputs)
        loss = float(_loss)
        y_true = []
        y_pred = []

        for i in self.val_step_outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
            
        y_pred = np.asanyarray(y_pred)#y_temp_pred y_pred
        y_true = np.asanyarray(y_true)
        
        pred_dict = {}
        pred_dict['y_pred']= y_pred
        pred_dict['y_true']= y_true
        
        val_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        
        self.log("val_acc", val_acc)
        # print("y_pred= ", y_pred)
        # print('\n\n\n')
        # print("y_true= ", y_true)
        # print('\n\n\n')
        print("-------val_report-------")
        metrics_dict = classification_report(y_true, y_pred,zero_division=1,
                                             target_names = self.label_names, 
                                             output_dict=True)
        df_result = pd.DataFrame(metrics_dict).transpose()
        pprint(df_result)
        
        
        df_result.to_csv(
            f'{Output_dir}/{datetime.now().__format__("%m%d_%H%M")}_DM_MM_{self.t_embed_type}_{self.a_embed_type}_val.csv')

        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(
            f'{Output_dir}/{datetime.now().__format__("%m%d_%H%M")}_DM_MM_{self.t_embed_type}_{self.a_embed_type}_val_pred.csv')
        self.val_step_outputs.clear()
        # self.val_step_targets.clear()
        return {'loss': _loss}

    def on_test_epoch_end(self):

        y_true = []
        y_pred = []

        for i in self.test_step_outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
            
        y_pred = np.asanyarray(y_pred)#y_temp_pred y_pred
        y_true = np.asanyarray(y_true)
        
        pred_dict = {}
        pred_dict['y_pred']= y_pred
        pred_dict['y_true']= y_true
        
        
        print("-------test_report-------")
        metrics_dict = classification_report(y_true, y_pred,zero_division=1,
                                             target_names = self.label_names, 
                                             output_dict=True)
        df_result = pd.DataFrame(metrics_dict).transpose()
        self.test_step_outputs.clear()
        # self.test_step_targets.clear()
        pprint(df_result)
        

        df_result.to_csv(
            f'{Output_dir}/{datetime.now().__format__("%m%d_%H%M")}_DM_MM_{self.t_embed_type}_{self.a_embed_type}_test.csv')

        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(
            f'{Output_dir}/{datetime.now().__format__("%m%d_%H%M")}_DM_MM_{self.t_embed_type}_{self.a_embed_type}_test_pred.csv')

    # def preprocess_existing_summary_dataframe():



def main(args,config):
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    seed_everything( config['random_seed'])
        
    model = Model(args,config) 
    # model.preprocess_dataframe()
    model.preprocess_loaded_summaries()
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=10,
        verbose=True,
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{SaveRoot}/Model/checkpoints",
        monitor='val_acc',
        auto_insert_metric_name=True,
        verbose=True,
        mode='max', 
        save_top_k=1,
      )    

    print(":: Start Training ::")
    #     
    trainer = Trainer(
        logger=False,
        callbacks=[early_stop_callback,checkpoint_callback],
        enable_checkpointing = True,
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        deterministic=True, # ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
        # For GPU Setup
        # gpus=[config['gpu']] if torch.cuda.is_available() else None,
        strategy='ddp_find_unused_parameters_true',
        precision=16 if args.fp16 else 32
    )
    trainer.fit(model)
    trainer.test(model,dataloaders=model.test_dataloader(),ckpt_path="best")
    
if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--random_seed", type=int, default=2023) 
    parser.add_argument("--t_embed", type=str, default="mbert") 
    parser.add_argument("--a_embed", type=str, default="en") 
    parser.add_argument("--SaveRoot", type=str, default='/mnt/External/Seagate/FedASR/LLaMa2/dacs') 
    parser.add_argument("--file_in", type=str, default='/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h_total.csv') 
    parser.add_argument("--process_summary", type=bool, default=False) 
    parser.add_argument("--summary_dir_in", type=str, default='/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-anomia') 
    
    config = parser.parse_args()
    SaveRoot=config.SaveRoot

    script_path, file_extension = os.path.splitext(__file__)

    # 使用os.path模組取得檔案名稱
    script_name = os.path.basename(script_path)

    Output_dir=f"{SaveRoot}/result/{script_name}/"
    os.makedirs(Output_dir, exist_ok=True)

    print(config)
    args = Arg()
    args.epochs=config.epochs
    args.t_hidden_size=Embsize_map['t_hidden_size'][config.t_embed]
    main(args,config.__dict__)       


"""

python 0207_DM_multi.py --gpu 1 --t_embed mbert --a_embed en
python 0207_DM_multi.py --gpu 1 --t_embed xlm --a_embed en

# don
python 0207_DM_multi.py --gpu 0 --t_embed xlm --a_embed gr
python 0207_DM_multi.py --gpu 1 --t_embed mbert --a_embed gr

"""