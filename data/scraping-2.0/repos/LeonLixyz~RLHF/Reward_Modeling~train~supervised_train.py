import sys
sys.path.append('/user/al4263/rlhf/Reward_Modeling')
import utils
from accelerate.utils import set_seed, broadcast
import openai
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
import numpy as np
import torch
import random
from dataclasses import dataclass, field
from transformers import HfArgumentParser


    
@dataclass
class ModelArguments:
    ref_size: int = field(default = None)
    num_ref_train: int = field(default = None)
    hidden_size: int = field(default = None)
    output_size: int = field(default = None)
    backbone_model: str = field(default = None)
    enn_gain: float = field(default=None)
    lmbda: float = field(default=None)
    reward_gain: float = field(default=1)

@dataclass
class TrainingArguments:
    lr: float = field(default=1e-5)
    enn_lr: float = field(default=1e-3)
    reward_lr: float = field(default=1e-3)
    num_epochs: int = field(default=1)
    eval_steps: int = field(default=10)
    eval_z_size_list: str = field(default = "10,100,1000")
    gradient_acc: int = field(default=1)
    warmup_ratio: float = field(default=0.03)
    enn_decay: float = field(default=0.5)
    weight_decay: float = field(default=0.01)
    reward_decay: float = field(default=0.01)
    seed: int = field(default=42)
    flash_attn: bool = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    
    def __post_init__(self):
        self.eval_z_size_list = [int(item.strip()) for item in self.eval_z_size_list.split(',')]

@dataclass
class DataArguments:
    dataset_name: str = field(default="alpaca_human_preference")
    train_batch_size: int = field(default=4)
    eval_batch_size: int = field(default=64)
    max_length: int = field(default=512)
    shuffle: bool = field(default=False)


@dataclass
class LoggingArguments:
    user: str = field(default="leon")
    project: str = field(default="reward-enn")
    wandb_project: str = field(default="lr_sweep")


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoggingArguments))
    model_args, training_args, data_args, logging_args = parser.parse_args_into_dataclasses()
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    os.environ["WANDB_MODE"] = "offline"
    torch.set_printoptions(precision=10)


    accelerator = Accelerator(log_with="wandb") 
    utils.generate_save_dir(model_args, training_args, data_args, logging_args)
    accelerator.print('wandb_dir',logging_args.wandb_dir)
    
    set_seed(training_args.seed)
    set_seed(training_args.seed + accelerator.process_index * 100)
    print('torch seed', torch.initial_seed())
    print('cuda seed', torch.cuda.initial_seed())

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.backbone_model,
        padding_side="left",
        )
    tokenizer.pad_token = tokenizer.eos_token


    training_dataloader, eval_dataloader, joint_eval_dataloader = utils.load_supervised_data(data_args, tokenizer)
    # print dataset sizes:
    accelerator.print('training dataset size:', len(training_dataloader))
    accelerator.print('eval dataset size:', len(eval_dataloader))
    accelerator.print('joint eval dataset size:', len(joint_eval_dataloader))

    model, optimizer = utils.create_new_model_and_optimizer(model_args, training_args, logging_args)
    scheduler = utils.get_scheduler(optimizer, training_args, data_args, training_dataloader, accelerator)

    model, optimizer, training_dataloader, eval_dataloader, joint_eval_dataloader, scheduler = accelerator.prepare(
            model, optimizer, training_dataloader, eval_dataloader, joint_eval_dataloader, scheduler
        )
    

    accelerator.print('--------------------------------------start training--------------------------------------')
    
    model, metric_log, reward_log = utils.supervised_trainer(model, optimizer, scheduler, training_dataloader, eval_dataloader, joint_eval_dataloader, accelerator, logging_args, training_args, model_args)

    if accelerator.is_main_process:
        utils.save_model(logging_args, model)
        # utils.plot_training_log(metric_log, reward_log, logging_args)
    
if __name__ == "__main__":
    main()