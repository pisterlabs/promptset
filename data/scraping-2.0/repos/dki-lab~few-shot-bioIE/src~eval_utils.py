import pickle
import pandas as pd
import os
import openai
import numpy as np
import ipdb
import re
import tqdm
import conlleval

import spacy
from data_utils import *
from scipy import special
import wandb
import json
from sklearn.metrics import precision_recall_fscore_support

nlp = spacy.load("en_core_web_sm")

def filter_by_dict(df, filter_v):
    for k,v in filter_v.items():
        df = df[df[k] == v]
    return df

def save_plm_metric_table(output_dir, training_args, eval_name='dev'):
    results_by_epoch = []

    # Aggregate results and compute metric
    for epoch in range(wandb.config.num_train_epochs // wandb.config.epoch_eval_period):
        epoch += 1  # Starts at
        epoch *= wandb.config.epoch_eval_period
        metrics = json.load(open(output_dir + '/{}.metrics.{}_results.json'.format(eval_name, epoch)))
        f1, precision, recall = metrics['{}_f1'.format(eval_name)], metrics['{}_precision'.format(eval_name)], metrics['{}_recall'.format(eval_name)]

        results_by_epoch.append((f1, precision, recall,
                                 wandb.run.id,
                                 epoch,
                                 training_args.per_device_train_batch_size,
                                 training_args.learning_rate,
                                 training_args.num_train_epochs,
                                 wandb.config.model_name,
                                 training_args.weight_decay,
                                 training_args.warmup_ratio
                                 ))

    results_by_epoch = pd.DataFrame(results_by_epoch, columns=['f1',
                                                               'precision',
                                                               'recall',
                                                               'run_id',
                                                               'epoch',
                                                               'batch_size',
                                                               'learning_rate',
                                                               'num_train_epochs',
                                                               'model_name',
                                                               'weight_decay',
                                                               'warmup_ratio'
                                                               ]
                                    )

    results_by_epoch.to_csv(output_dir + '/{}.metrics'.format(eval_name))

def create_bio_preds(df, pred_name, output_col_name='bio_preds'):
    """Function to create a BIO Tag from GPT-3 Predicted entities"""

    bio_preds = []
    post_processed_ents_col = []

    for i, row in df.iterrows():
        try:
            sent = token_preprocessing(row['orig_tok_sent'].lower())
        except:
            sent = token_preprocessing(row['sents'].lower())

        # bio_tags = row['ner_seq']
        predicted_entities = [p.strip().lower() for p in row[pred_name]]

        post_predicted_ents = post_processing(sent, predicted_entities)
        post_processed_ents_col.append(post_predicted_ents[:])

        # Sort by Length, Longest to Shortest
        pred_ent_inds_by_length = np.argsort([len(e) for e in post_predicted_ents], kind='mergesort')[::-1]
        post_predicted_ents = np.array(post_predicted_ents)[pred_ent_inds_by_length]

        bio_pred_seq = ' ' + sent + ' '

        for pred_ent in post_predicted_ents:

            pred_ent = token_preprocessing(' '.join([s.text for s in nlp.tokenizer(str(pred_ent))]))
            pred_ent = re.sub('\s+', ' ', pred_ent)

            if pred_ent != '':
                pred_bios = ['I|||' for _ in pred_ent.split()]

                pred_bios[0] = 'B|||'

                pred_bios = ' '.join(pred_bios)

                bio_pred_seq = bio_pred_seq.replace(' ' + pred_ent + ' ', ' ' + pred_bios + ' ')

        bio_pred_seq = ' '.join(['O' if (w != 'B|||' and w != 'I|||') else w for w in bio_pred_seq.split()])
        bio_pred_seq = bio_pred_seq.replace('|', '')
        bio_pred_seq = bio_pred_seq.strip()

        # assert len(bio_tags.split()) == len(bio_pred_seq.split()), ipdb.set_trace()
        bio_preds.append(bio_pred_seq)

    df[output_col_name] = bio_preds
    df[pred_name + '.post'] = post_processed_ents_col

    return df

def post_processing(sentence, predicted_ents):
    post_predicted_ents = []

    for ent in predicted_ents:
        ent = ent.replace(',', '')

        # Tokenizing generated text in same way as original dataset
        ent = token_preprocessing(' '.join([s.text for s in nlp.tokenizer(str(ent))]))

        # Removing phrases which are not standalone in sentence
        if ' ' + ent + ' ' in ' ' + token_preprocessing(sentence).lower() + ' ':
            post_predicted_ents.append(ent)

    return list(set(post_predicted_ents))

def conlleval_eval(true, preds, verbose=False):
    true = [[t[0] + '-X' for t in s.split()] for s in true]
    preds = [[t[0] + '-X' for t in s.split()] for s in preds]
    true = np.concatenate(true)
    preds = np.concatenate(preds)

    prec, recall, f1 = conlleval.evaluate(true, preds, verbose=verbose)

    return f1, prec, recall

def evaluate_gpt3_output(df, task, params, prediction_name='predictions'):
    if task == 'NER':

        #Populating BIO Tags based on entity names like GPT-3 for fair comparison
        sep = params['sep']
        df['full_entities'] = [sep.join(e).split(sep) for e in df['entities']]
        df_ = create_bio_preds(copy.deepcopy(df), 'full_entities', 'ner_seq')

        #Populating BIO Tag predictions from entity names generated
        df_ = create_bio_preds(df_, prediction_name)

        different_labels = df_[df['ner_seq'] != df_['ner_seq']]
        if len(different_labels) > 0:
            pass#ipdb.set_trace()
        df = df_
        f1, precision, recall = conlleval_eval(df.ner_seq, df.bio_preds)
    elif task == 'RE':
        precision, recall, f1, support = precision_recall_fscore_support(y_pred=df[prediction_name], y_true=df['verbalized_label'],
                                                     labels=[params['label_verbalizer'][l] for l in params['pos_labels']],
                                                     average='micro')

    cost = df.cost.sum()
    time_spent = df.time.sum()
    return {'f1': f1, 'precision': precision, 'recall': recall,'total_cost':cost, 'total_time': time_spent}