from BertAdam import BertAdam
import torch
from optimization_openai import OpenAIAdam
from transformers import AdamW


def ComputeBertAdamOptimizer(model, num_train_optimization_steps):
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=5e-5,
                         warmup=0.1,
                         t_total=num_train_optimization_steps)
    return optimizer


def ComputeAdamOptimizer(model, num_train_optimization_steps, lr=5e-5):
    return torch.optim.Adam(model.parameters(), lr=lr)


def ComputeAdamWOptimizer(model, num_train_optimization_steps, lr=5e-5):
    # print('\n', lr, '\n')
    # return AdamW(model.parameters(), lr=1e-4)
    return AdamW(model.parameters(), lr=lr, eps=1e-6)


def ComputerOpenAIOptimizer(model, num_train_optimization_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=6.25e-5,
                           warmup=0.002,
                           max_grad_norm=1,
                           weight_decay=0.01,
                           t_total=num_train_optimization_steps)
    return optimizer
