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
from transformers import AutoTokenizer, AutoModel
from prompts import assesmentPrompt_template, Instruction_templates, Psychology_template,\
    Sensitive_replace_dict, generate_psychology_prompt
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, FewShotPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from addict import Dict
import librosa

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
Embsize_map={}
Embsize_map['t_hidden_size']={'xlm':1280,'mbert':768,'Semb':1536,'Embedding':350}

Audio_pretrain=set(['en','gr','multi','wv'])
xlm_mlm_100_1280=set(['xlm_sentence','xlm_session'])
mbert_series=set(['mbert_sentence','mbert_session',])
bert_library=set(["bert-base-uncased",
                "xlm-roberta-base",
                "albert-base-v1",
                "xlnet-base-cased",
                "emilyalsentzer/Bio_ClinicalBERT",
                "dmis-lab/biobert-base-cased-v1.2",
                "YituTech/conv-bert-base",])
Text_Summary=set(['anomia'])

Text_pretrain = xlm_mlm_100_1280 | mbert_series | bert_library


# ,'Semb','Embedding'
Model_settings_dict=Dict()
Model_settings_dict['mbert_sentence']={
    'inp_col_name': 'text',
    'inp_hidden_size': 768,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['mbert_session']={
    'inp_col_name': 'text',
    'inp_hidden_size': 768,
    'file_in':"/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-anomia",
}
Model_settings_dict['xlm_sentence']={
    'inp_col_name': 'text',
    'inp_hidden_size': 1280,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['xlm_session']={
    'inp_col_name': 'text',
    'inp_hidden_size': 1280,
    'file_in':"/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-anomia",
}
Model_settings_dict['en']={
    'inp_col_name': 'path',
    'inp_hidden_size': 512,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['gr']={
    'inp_col_name': 'path',
    'inp_hidden_size': 512,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['multi']={
    'inp_col_name': 'path',
    'inp_hidden_size': 512,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['wv']={
    'inp_col_name': 'path',
    'inp_hidden_size': 512,
    'file_in':'/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
}
Model_settings_dict['anomia']={
    'inp_col_name': 'Psych_Summary',
    'inp_hidden_size': Embsize_map['t_hidden_size']['mbert'], # use mbert to tokenize and model it
    'file_in':'/mnt/External/Seagate/FedASR/LLaMa2/dacs/EmbFeats/Lexical/Embeddings/text_data2vec-audio-large-960h_Phych-anomia',
}

model_settings_dict = {
    'bert-base-uncased': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'xlm-roberta-base': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'albert-base-v1': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'xlnet-base-cased': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'emilyalsentzer/Bio_ClinicalBERT': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'dmis-lab/biobert-base-cased-v1.2': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
    'YituTech/conv-bert-base': {
        'inp_col_name': 'text',
        'inp_hidden_size': 768,
        'file_in': '/home/FedASR/dacs/centralized/saves/results/data2vec-audio-large-960h',
    },
}

Model_settings_dict.update(model_settings_dict)


def check_keys_matching(*sets, model_settings_dict):
    # Combine all input sets using union
    all_pretrain = set().union(*sets)
    
    # Get the keys from Model_settings_dict
    model_keys = set(model_settings_dict.keys())
    
    # Check if all keys in Model_settings_dict match the combined pretrain set
    return model_keys == all_pretrain

# Example usage
assert check_keys_matching(Audio_pretrain, Text_pretrain, Text_Summary, model_settings_dict=Model_settings_dict)



class ModelArg:
    version = 1
    # data
    epochs: int = 5  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 350  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    batch_size: int = 8

class SingleForwardModel(LightningModule):
    # """
    # Is the model of 0207_DM_SentenceLvl1input.py
    # """
    def __init__(self, args,config):
        super().__init__()
        # config:
        
        self.mdlArg = args.mdlArg
        self.args=args
        self.config = config
        self.batch_size = self.mdlArg.batch_size
        
        # meta data:
        self.epochs_index = 0
        self.label_cols = 'dementia_labels'
        self.label_names = ['Control','ProbableAD']
        self.num_labels = 2
        
        
        # --> HERE STEP 1 <--
        # ATTRIBUTES TO SAVE BATCH OUTPUTS
        self.test_step_outputs = []   # save outputs in each batch to compute metric overall epoch
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch

        
        
        
        # Variables that may vary
        # self.inpArg = args.inpArg
        # self.inp_embed_type = self.config['inp_embed']
        # self.inp_col_name = self.inpArg.inp_col_name
        # self.inp_hidden_size = self.inpArg.inp_hidden_size
        # self.hidden = int(self.inpArg.linear_hidden_size)
        # self.inp_tokenizer, self.inp_model, self.pooler=self._setup_embedding(self.inp_embed_type, self.inp_hidden_size)
        # self.clf1 = nn.Linear(self.hidden, int(self.hidden/2))
        # self.clf2 = nn.Linear(int(self.hidden/2), self.num_labels)

    def _setup_embedding(self,inp_embed_type, inp_hidden_size):
        if inp_embed_type == "en":
            inp_pretrained = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            inp_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(inp_pretrained)
            inp_model = Wav2Vec2Model.from_pretrained(inp_pretrained)
        elif inp_embed_type == "gr":
            inp_pretrained = "lighteternal/wav2vec2-large-xlsr-53-greek"
            inp_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(inp_pretrained)
            inp_model = Wav2Vec2Model.from_pretrained(inp_pretrained)
        elif inp_embed_type == "multi":
            inp_pretrained = "voidful/wav2vec2-xlsr-multilingual-56"
            inp_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(inp_pretrained)
            inp_model = Wav2Vec2Model.from_pretrained(inp_pretrained)
        elif inp_embed_type == "wv":
            inp_pretrained = 'facebook/wav2vec2-base'
            inp_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(inp_pretrained)
            inp_model = Wav2Vec2Model.from_pretrained(inp_pretrained)
        elif inp_embed_type in mbert_series:
            inp_pretrained = 'bert-base-multilingual-uncased'
            inp_tokenizer = BertTokenizer.from_pretrained(inp_pretrained)
            inp_model = BertModel.from_pretrained(inp_pretrained)
        elif inp_embed_type in xlm_mlm_100_1280:
            inp_pretrained = 'xlm-mlm-100-1280'
            inp_tokenizer = XLMTokenizer.from_pretrained(inp_pretrained)
            inp_model = XLMModel.from_pretrained(inp_pretrained)
        elif inp_embed_type in Text_Summary:
            inp_pretrained = 'bert-base-multilingual-uncased'
            inp_tokenizer = BertTokenizer.from_pretrained(inp_pretrained)
            inp_model = BertModel.from_pretrained(inp_pretrained)
        elif inp_embed_type in bert_library:
            inp_pretrained = inp_embed_type
            inp_tokenizer = AutoTokenizer.from_pretrained(inp_pretrained)
            inp_model = AutoModel.from_pretrained(inp_pretrained)
        else:
            raise ValueError(f"{inp_embed_type} seems not in Model_settings_dict")
        pooler = BertPooler(inp_hidden_size)
        return inp_tokenizer, inp_model, pooler
    def _get_embedding(self,inp,inp_embed_type, inp_model, pooler):
        if inp_embed_type in mbert_series:
            out = inp_model(inp)[1]
        elif inp_embed_type in xlm_mlm_100_1280:
            out = inp_model(inp)[0]
            out = pooler(out)
        elif inp_embed_type in bert_library:
            if inp_embed_type=='xlnet-base-cased':
                out = inp_model(inp)[0]
                out = pooler(out)
            else:
                out = inp_model(inp)[1]
        elif inp_embed_type in Audio_pretrain:
            out = inp_model(inp)['extract_features']  # [2] # last_hidden_state, feature extraction
            out = out[:, 0, :]
        elif inp_embed_type in Text_Summary:
            out = inp_model(inp)[1]
        else:
            raise ValueError("Invalid inp_embed_type specified.")
        
        return out
    def forward(self, inp):
        output = self._get_embedding(inp,self.inp_embed_type, self.inp_model, self.pooler)
        
        logits = self.clf2(self.clf1(output))
    
        return logits
        

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def _Tokenize(self, df, inp_embed_type, inp_col_name, inp_tokenizer,tg_sr=16000):
        if inp_embed_type in Text_pretrain:
            df[inp_col_name] = df[inp_col_name].map(lambda x: inp_tokenizer.encode(
                str(x),
                padding='max_length',
                max_length=self.mdlArg.max_length,
                truncation=True,
            ))
        elif inp_embed_type in Text_Summary:
            df[inp_col_name] = df[inp_col_name].map(lambda x: inp_tokenizer.encode(
                str(x),
                padding='max_length',
                max_length=self.mdlArg.max_length,
                truncation=True,
            ))
        elif inp_embed_type in Audio_pretrain:
            audio_root = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/clips"
            df[inp_col_name] = df[inp_col_name].map(lambda x: inp_tokenizer(
                librosa.load(f"{audio_root}/{x}")[0],
                padding='max_length',
                sampling_rate=tg_sr,
                max_length=100000,
                truncation=True
            )['input_values'][0])
        else:
            raise ValueError("Invalid inp_embed_type specified.")
        return df
    def preprocess_dataframe(self):
        
        df_train = pd.read_csv(f"{self.inpArg.file_in}/train.csv")
        df_dev = pd.read_csv(f"{self.inpArg.file_in}/dev.csv")
        df_test = pd.read_csv(f"{self.inpArg.file_in}/test.csv")
        self.df_train=self._Tokenize(df_train, self.inp_embed_type, self.inpArg.inp_col_name,self.inp_tokenizer)
        self.df_dev=self._Tokenize(df_dev, self.inp_embed_type,self.inpArg.inp_col_name,self.inp_tokenizer)
        self.df_test=self._Tokenize(df_test, self.inp_embed_type,self.inpArg.inp_col_name,self.inp_tokenizer)

        print(f'# of train:{len(df_train)}, val:{len(df_dev)}, test:{len(df_test)}')
        self._df2Dataset()
    def _df2Dataset(self):
        def DecideDtype(inp_embed_type):
            if inp_embed_type in Text_pretrain:
                dtype=torch.long
            elif inp_embed_type in Audio_pretrain:
                dtype=torch.float
            return dtype
        dtype=DecideDtype(self.inp_embed_type)
        self.train_data = TensorDataset(
            torch.tensor(self.df_train[self.inpArg.inp_col_name].tolist(), dtype=dtype),
            torch.tensor(self.df_train[self.label_cols].tolist(), dtype=torch.long),
        )
        
        self.val_data = TensorDataset(
             torch.tensor(self.df_dev[self.inpArg.inp_col_name].tolist(), dtype=dtype),
            torch.tensor(self.df_dev[self.label_cols].tolist(), dtype=torch.long),
        )

        self.test_data = TensorDataset(
             torch.tensor(self.df_test[self.inpArg.inp_col_name].tolist(), dtype=dtype),
            torch.tensor(self.df_test[self.label_cols].tolist(), dtype=torch.long),
             torch.tensor(self.df_test.index.tolist(), dtype=torch.long),
        )

    def train_dataloader(self):
        
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.mdlArg.cpu_workers,
        )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.mdlArg.cpu_workers,
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.mdlArg.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        # token, audio, labels = batch  
        token,  labels = batch  
        # logits = self(token, audio) 
        logits = self(token) 
        loss = nn.CrossEntropyLoss()(logits, labels)   
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        # token, audio, labels = batch  
        token, labels = batch  
        # logits = self(token, audio) 
        logits = self(token) 
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
        # token, audio, labels,id_ = batch 
        token, labels,id_ = batch 
        print('id', id_)
        # logits = self(token, audio) 
        logits = self(token) 
        
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
        

        # df_result.to_csv(
        #     f'{self.args.Output_dir}/{self.inp_embed_type}_val.csv')

        # pred_df = pd.DataFrame(pred_dict)
        # pred_df.to_csv(
        #     f'{self.args.Output_dir}/{self.inp_embed_type}_val_pred.csv')
        self._save_results_to_csv(df_result, pred_dict, self.args, suffix='_val')
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
        

        # df_result.to_csv(
        #     f'{self.args.Output_dir}/{self.inp_embed_type}_test.csv')

        # pred_df = pd.DataFrame(pred_dict)
        # pred_df.to_csv(
        #     f'{self.args.Output_dir}/{self.inp_embed_type}_test_pred.csv')
        self._save_results_to_csv(df_result, pred_dict, self.args, suffix='_test')
    def _DecideDtype(self,inp_embed_type):
        if inp_embed_type in Text_pretrain:
            dtype=torch.long
        elif inp_embed_type in Audio_pretrain:
            dtype=torch.float
        elif inp_embed_type in Text_Summary:
            dtype=torch.long
        return dtype
    def _safe_output(self):
        self.outStr=self.inp_embed_type.replace("/","__")
    def _save_results_to_csv(self, df_result, pred_dict, args, suffix):
        # Save df_result to CSV
        self._safe_output()
        df_result.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}.csv')

        # Save pred_df to CSV
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(f'{args.Output_dir}/{self.outStr}{suffix}_pred.csv')


    