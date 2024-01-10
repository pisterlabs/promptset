from optimization_openai import OpenAIAdam
import math
import time
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch
import torch.nn as nn
import optimizers
from train_process import *
import os
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from DataBunch import *
from network import BigModel
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from Optim import ScheduledOptim
from tqdm import tqdm
from Config_File import Config_base
import argparse
import datetime
import sys
import time

parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('--config', type=str, default='Config_base')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--causal_ratio', type=float,
                    default=-0.001, help='batch_size')
parser.add_argument('--learning_rate', type=float,
                    default=3e-5, help='lr')
parser.add_argument('--batch_size_test', type=int,
                    default=None, help='batch_size_test')
parser.add_argument('--epoch', type=int, default=None, help='epoch')
parser.add_argument('--dataset', type=str, default='mini-SNLI', help='dataset')
parser.add_argument('--grad_loss_func', type=str,
                    default='argmax_loss', help='grad_loss_func')
parser.add_argument('--saliancy_method', type=str,
                    default='compute_saliancy_batch', help='saliancy_method')
parser.add_argument('--train_process', type=str,
                    default='train', help='train_process')
parser.add_argument('--model_name_or_path', type=str,
                    default='bert-base-uncased', help='model_name_or_path')
parser.add_argument('--databunch_method', type=str,
                    default='DataBunch', help='databunch_method')
parser.add_argument('--use_custom_bert', action='store_true',
                    default=False, help='use_custom_bert')
#parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='model_name_or_path')
parser.add_argument('--load_few', action='store_true',
                    default=False, help='load few')
parser.add_argument('--grad_clamp', action='store_true',
                    default=False, help='grad_clamp')
parser.add_argument('--test_mode', action='store_true',
                    default=False, help='test_mode or not')
parser.add_argument('--no_use_pre_train_parameters', action='store_true', default=False,
                    help='no_use_pre_train_parameters or not')

args = parser.parse_args()
print('\n', args, '\n', flush=True)
# Config_File.ComputeConfig(args.config)
#Config_File.Config = Config_File.ComputeConfig(args)

config = Config_base(args)

save_path = '../log/{}_{}_{}'.format(config.model_name_or_path, config.batch_size, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# output to txt 
class Logger(object):
    def __init__(self, filename=save_path+'.txt', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

sys.stdout = Logger(save_path+'.txt', sys.stdout)
# from train_augment_process import *
# from train_process_MM import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True


# from Augmentation import Analogy_Auger
# from train_process_MM import BigModel_two_bert

params_trainset = {'batch_size': config.batch_size,
                   'shuffle': False,
                   'num_workers': 0}
# print(config.batch_size_test)
# exit()
params_testset = {'batch_size': config.batch_size_test,
                  'shuffle': False,
                  'num_workers': 0}
torch.set_printoptions(threshold=10000)


def predict(model, test_generator, outputFile):
    pred = evaluate(model, criterion, test_generator, False)
    with open(outputFile, 'w', encoding='utf-8') as writer:
        writer.write('Id,Expected\n')
        for term in pred:
            pred_ret = term[1]
            pred_label = None
            if config.dataset_train == 'RTE':
                if pred_ret == 0:
                    pred_label = 'not_entailment'
                else:
                    pred_label = 'entailment'
            if config.dataset_train == 'MSRP':
                pred_label = pred_ret
            if config.dataset_train in ['SNLI', 'mini-SNLI', 'e-snli']:
                if pred_ret == 0:
                    pred_label = 'neutral'
                elif pred_ret == 1:
                    pred_label = 'entailment'
                else:
                    pred_label = 'contradiction'

            writer.write('{},{}\n'.format(term[0], pred_label))
        writer.close()


record = {}


def ReadDataset(args):
    global trainset, testset, devset
    db_class = globals()[config.databunch_method]
    dataset = config.dataset
    if config.do_train:
        trainset[dataset] = db_class(config, config.train_file_dict[dataset], config.sent_token_dict[dataset], config.label_token_dict[dataset],
                                     config.tokenizer, config.sent2_token_dict[dataset],
                                     dataset=dataset, id_token=config.id_token_dict[dataset], load_few=args.load_few)
        devset[dataset] = db_class(config, config.dev_file_dict[dataset], config.sent_token_dict[dataset],
                                   config.label_token_dict[dataset], config.tokenizer, config.sent2_token_dict[dataset],
                                   dataset=dataset, id_token=config.id_token_dict[dataset])
    if config.do_test:
        testset[dataset] = db_class(config, config.test_file_dict[dataset], config.sent_token_dict[dataset],
                                    None, config.tokenizer, config.sent2_token_dict[dataset], dataset=dataset, id_token=config.id_token_dict[dataset])


if __name__ == "__main__":

    trainset = {}
    testset = {}
    devset = {}
    ReadDataset(config)

    mode_result = []
    # print(config.__dict__)

    # model = network.__dict__[Config.big_model](w2id[Config.dataset_train]).to(Config.device)
    model = globals()[config.big_model](config)
    # print(model)
    # exit()
    # print(model.state_dict()['pre_emb_model.embeddings.word_embeddings.weight'])
    # for name, p in model.named_parameters():
    #     print(name)
    #     print(p)
    # exit()
    if config.multiple_gpu:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.to(config.device)
    if config.continue_train:
        model.load_state_dict(torch.load(config.model_save_path))
        print('continue the training model in {}'.format(config.model_save_path))

    criterion = nn.CrossEntropyLoss()

    if config.do_train:
        train_generator = data.DataLoader(
            trainset[config.dataset_train], **params_trainset)
        dev_generator = data.DataLoader(
            devset[config.dataset_train], **params_testset)

        optimizer = optimizers.__dict__[config.optimizer](model, int(
            len(trainset[config.dataset_train]) / params_trainset['batch_size']) * config.epoch, lr=args.learning_rate)
        # print(args.learning_rate)
        # exit()
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=int(len(train_generator) * config.epoch))

        mode_result.append(
            config.config_name + '\t' + str(globals()[args.train_process](config, model, optimizer, scheduler, criterion, train_generator, dev_generator)))
        for mode in mode_result:
            print(mode)
    if config.do_test:
        model.load_state_dict(torch.load(config.model_save_path))
        test_generator = data.DataLoader(
            testset[config.dataset_test], **params_testset)
        predict(model, test_generator, config.output_test_file)
