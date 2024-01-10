"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig
from model import MemoryGPT as GPT
from my_configuration_roberta import MemoryRobertaConfig
from my_modeling_roberta import MemoryRobertaModel

from my_utils import get_seq_train_batch

# -----------------------------------------------------------------------------
# default config values designed to train a evolver (roberta)
evolver_n_layer = 6
evolver_n_head = 12
evolver_n_embd = 768
evolver_n_intermediate = 3072
evolver_n_mem = 50

######### no use
evolver_pad_token_id = 0
evolver_gpt2_token_id_offset = 20 # the token id produced by gpt2 tokenizer should added by this offset
#################

segment_num = 1 # if > 1, train memory
num_target_model_layer = 12

remember_prob = 95 # 大于这个的话，minibatch的记忆都会被删除
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
wandb_notes=''
seed=1337
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
pretrained_model_name = 'gpt2'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
min_block_size = 50
# model
load_name = 'place_holder'
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    
    # 多此一举
    # # world_size number of processes will be training simultaneously, so we can scale
    # # down the desired gradient accumulation iterations per process proportionally
    # assert gradient_accumulation_steps % ddp_world_size == 0
    # gradient_accumulation_steps //= ddp_world_size
else:
    ddp_rank = 0
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# get initial data pointer for each batch in this rank: this_rank_batch_train_data_pointer
this_rank_train_data_start = ddp_rank * (len(train_data) // ddp_world_size)

if ddp_rank == (ddp_world_size-1):
    this_rank_train_data_end = len(train_data)
else:
    this_rank_train_data_end = (ddp_rank + 1) * (len(train_data) // ddp_world_size)

this_rank_data_num = this_rank_train_data_end - this_rank_train_data_start
this_rank_batch_train_data_pointer = []
actual_batch_size = batch_size * gradient_accumulation_steps
for bi in range(0, actual_batch_size):
    this_rank_batch_train_data_pointer.append(this_rank_train_data_start + bi * (this_rank_data_num // actual_batch_size))

# --------------------------------------------------------------------------
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# load pretrained model
if "gpt" in pretrained_model_name:
    print(f"Initializing from OpenAI GPT-2 weights: {pretrained_model_name}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    pretrained_model = GPT.from_pretrained(pretrained_model_name, override_args)
    pretrained_model.to(device)
    # backbone forzen
    for p in pretrained_model.parameters():
        p.requires_grad_(False)
    
    pretrained_model_config = pretrained_model.config
else:
    raise Exception(f"Unrecognized pretrained model {pretrained_model_name}")

# --------------------------------------------------------------------------
# create my evolver 
if init_from == 'scratch':
    evolver_config = MemoryRobertaConfig(vocab_size=pretrained_model_config.vocab_size + evolver_gpt2_token_id_offset, num_hidden_layers=evolver_n_layer,
                                        num_attention_heads=evolver_n_head, hidden_size=evolver_n_embd, max_position_embeddings=block_size, intermediate_size=evolver_n_intermediate,
                                        pad_token_id=evolver_pad_token_id, gpt2_token_id_offset=evolver_gpt2_token_id_offset, num_memory=evolver_n_mem,
                                        num_target_model_layer=num_target_model_layer, no_embeddings=True)
    evolver_model = MemoryRobertaModel(evolver_config)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint. 
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_config = checkpoint['evolver_config']
    # create the model
    evolver_model = MemoryRobertaModel(checkpoint_config)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    evolver_model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from == 'load':
    evolver_config = MemoryRobertaConfig(vocab_size=pretrained_model_config.vocab_size + evolver_gpt2_token_id_offset, num_hidden_layers=evolver_n_layer,
                                        num_attention_heads=evolver_n_head, hidden_size=evolver_n_embd, max_position_embeddings=block_size, intermediate_size=evolver_n_intermediate,
                                        pad_token_id=evolver_pad_token_id, gpt2_token_id_offset=evolver_gpt2_token_id_offset, num_memory=evolver_n_mem,
                                        num_target_model_layer=num_target_model_layer, no_embeddings=True)
    evolver_model = MemoryRobertaModel(evolver_config)

    # resume training from a checkpoint. 
    ckpt_path = os.path.join(out_dir, load_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['evolver_model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    evolver_model.load_state_dict(state_dict)
else:
    raise Exception(f"Unrecognized init_from {init_from}")

evolver_model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = evolver_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    pretrained_model_config = torch.compile(pretrained_model_config) # requires PyTorch 2.0
    evolver_model = torch.compile(evolver_model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
    # model = DDP(model, device_ids=[ddp_local_rank]) 
    evolver_model = DDP(evolver_model, device_ids=[ddp_local_rank], broadcast_buffers=False, find_unused_parameters=False) # https://github.com/pytorch/pytorch/issues/22095

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_repeat_loss():
    out = {}
    evolver_model.eval()
    for split in ['train', 'val']:
        if split == 'train':
            data = train_data
        elif split == 'val':
            data = val_data
        else:
            raise NotImplementedError
        
        losses = torch.zeros(eval_iters)
        gpt_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            this_iter_loss = torch.tensor(0.0, device=device)
            this_iter_gpt_loss = torch.tensor(0.0, device=device)

            # random start
            random_data_start_pointer = []
            for _ in range(0, actual_batch_size):
                random_data_start_pointer.append(random.randint(0, len(data) - block_size * segment_num - 1))

            # fetch data for this batch
            random_data_start_pointer, X, Y, attention_mask, seg_length_list = get_seq_train_batch(data, random_data_start_pointer, segment_num, block_size, min_block_size, device, device_type) # fetch the very first batch

            # empty memory
            input_memory_list = [None for _ in range(gradient_accumulation_steps)]

            for micro_step in range(gradient_accumulation_steps):
                this_micro_X = X[batch_size*micro_step : batch_size*(1+micro_step)]
                this_micro_Y = Y[batch_size*micro_step : batch_size*(1+micro_step)]
                this_micro_attention_mask = attention_mask[batch_size*micro_step : batch_size*(1+micro_step)]
                this_micro_seg_length_list = seg_length_list[:, batch_size*micro_step : batch_size*(1+micro_step)] # (seg num, micro batch size)

                # get memory of last step
                input_memory = input_memory_list[micro_step]

                target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)
                
                for si in range(segment_num):
                    # get data
                    this_x = this_micro_X[:, si, :]
                    this_y = this_micro_Y[:, si, :]
                    this_attention_mask = this_micro_attention_mask[:, si, :]
                    this_seg_len = this_micro_seg_length_list[si]

                    # ----------------------------------------------
                    output_embeds = pretrained_model(idx=this_x, input_parameter=target_model_parameter, output_embeds=True)

                    # X -> memory
                    input_memory = evolver_model(inputs_embeds=output_embeds, attention_mask=this_attention_mask, input_memory=input_memory)["memory_output"]

                    # last memory -> X
                    target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)
                    # ----------------------------------------------

                    # predict next segment with memory
                    _, loss = pretrained_model(this_x, this_y, target_model_parameter)
                    this_iter_loss = loss + this_iter_loss

                    _, gpt_loss  = pretrained_model(this_x, this_y)
                    this_iter_gpt_loss = gpt_loss + this_iter_gpt_loss

            this_iter_loss = this_iter_loss / (gradient_accumulation_steps * segment_num)
            this_iter_gpt_loss = this_iter_gpt_loss / (gradient_accumulation_steps * segment_num)

            losses[k] = this_iter_loss.item()
            gpt_losses[k] = this_iter_gpt_loss.item()
        out[split] = losses.mean()
        out[split + "_gpt"] = gpt_losses.mean()
    evolver_model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, notes=wandb_notes)

# training loop
this_rank_batch_train_data_pointer, X, Y, attention_mask, seg_length_list = get_seq_train_batch(train_data, this_rank_batch_train_data_pointer, segment_num, block_size, min_block_size, device, device_type) # fetch the very first batch

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
# raw_model = model.module if ddp else model # unwrap DDP container if needed
raw_evolver_model = evolver_model.module if ddp else evolver_model # unwrap DDP container if needed
running_mfu = -1.0

input_memory_list = [None for _ in range(gradient_accumulation_steps)]

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_repeat_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "val/train_loss": losses['train'],
                "val/val_loss": losses['val'],
                "val/train_gpt_loss": losses['train_gpt'],
                "val/val_gpt_loss": losses['val_gpt'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'evolver_model': raw_evolver_model.state_dict(),
                    'evolver_config': evolver_config,
                    'optimizer': optimizer.state_dict(),
                    'pretrained_model_config': pretrained_model_config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    
    lossf = 0.0
    gpt_lossf = 0.0

    repeat_lossf = 0.0
    revise_lossf = 0.0

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        all_loss = torch.tensor(0.0, device=device, requires_grad=True)
        gpt_loss = torch.tensor(0.0, device=device)
        
        repeat_loss = torch.tensor(0.0, device=device)
        revise_loss = torch.tensor(0.0, device=device)
        
        input_memory = input_memory_list[micro_step]

        this_micro_X = X[batch_size*micro_step : batch_size*(1+micro_step)]
        this_micro_Y = Y[batch_size*micro_step : batch_size*(1+micro_step)]
        this_micro_attention_mask = attention_mask[batch_size*micro_step : batch_size*(1+micro_step)]

        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            evolver_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)

            past_segments_x = []
            past_segments_y = []

            # sampled_segments_index = set(random.sample(range(segment_num), max(int(segment_num * 0.5), 1)))
            # trained_seg_num = segment_num + len(sampled_segments_index)

            sampled_segments_index = set(range(segment_num))
            trained_seg_num = segment_num + len(sampled_segments_index)

            for si in range(segment_num):
                this_x = this_micro_X[:, si, :]
                this_y = this_micro_Y[:, si, :]
                this_attention_mask = this_micro_attention_mask[:, si, :]

                # 保存数据用于复习
                if si in sampled_segments_index:
                    past_segments_x.append(this_x)
                    past_segments_y.append(this_y)

                # generate embeddings by pretrained model
                with torch.no_grad():
                    output_embeds = pretrained_model(idx=this_x, input_parameter=target_model_parameter, output_embeds=True)

                # X -> memory
                input_memory = evolver_model(inputs_embeds=output_embeds, attention_mask=this_attention_mask, input_memory=input_memory)["memory_output"]

                target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)
                
                # last memory -> X
                _, loss = pretrained_model(this_x, this_y, target_model_parameter)
                all_loss = loss / trained_seg_num + all_loss
                repeat_loss = loss / segment_num + repeat_loss

                with torch.no_grad():
                    _, loss = pretrained_model(this_x, this_y)
                    gpt_loss = loss / trained_seg_num + gpt_loss

            # 复习一下past_segments
            for (this_x, this_y) in zip(past_segments_x, past_segments_y):
                _, loss = pretrained_model(this_x, this_y, target_model_parameter)
                all_loss = loss / trained_seg_num + all_loss
                revise_loss = loss / len(past_segments_x) + revise_loss

                with torch.no_grad():
                    _, loss = pretrained_model(this_x, this_y)
                    gpt_loss = loss / trained_seg_num + gpt_loss

            ### 
            all_loss = all_loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            gpt_loss = gpt_loss / gradient_accumulation_steps

            repeat_loss = repeat_loss / gradient_accumulation_steps
            revise_loss = revise_loss / gradient_accumulation_steps

        lossf += all_loss.item()
        gpt_lossf += gpt_loss.item()

        repeat_lossf += repeat_loss.item()
        revise_lossf += revise_loss.item()

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        this_rank_batch_train_data_pointer, X, Y, attention_mask, seg_length_list = get_seq_train_batch(train_data, this_rank_batch_train_data_pointer, segment_num, block_size, min_block_size, device, device_type)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(all_loss).backward()

        # input_memory_list[micro_step] = input_memory.detach()
        input_memory_list[micro_step] = None

        # # 重启记忆
        # for bi in range(input_memory.shape[0]):
        #     if random.randint(1, 100) > remember_prob:
        #         input_memory_list[micro_step][bi] = raw_evolver_model.initial_memory

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(evolver_model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = evolver_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, gpt_loss {gpt_lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "train/gpt_loss": gpt_lossf,
                "train/repeat_loss": repeat_lossf,
                "train/revise_loss": revise_lossf,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
