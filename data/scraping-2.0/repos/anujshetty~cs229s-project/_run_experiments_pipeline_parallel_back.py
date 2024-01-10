import os
import time
import math
import json
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F 
from torch.profiler import profile, record_function
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, broadcast, barrier, send, recv

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 5
log_interval = 1
eval_iters = 1
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'gpt2-medium'#'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'cs229s'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'wikitext' #'shakespeare'
gradient_accumulation_steps = 2 #5 * 8 # used to simulate larger batch sizes
batch_size = 4 #2 #11 # if gradient_accumulation_steps > 1, this is the micro-batch size
batch_split_size = 4
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 1 #5 # total number of training iterations
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
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
actual_ddp = False # ddp is used for other checks, this new flag is for data parallelism specifically, separate from pipeline parallelism
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
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
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
def init_model(block_size):
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        if master_process:
            print(f"Initializing a new model from scratch : block_size {block_size}, batch_size {batch_size}")
            print(f"Batch size {batch_size} gradient acc steps {gradient_accumulation_steps} batch split size {batch_split_size}")
            # determine the vocab size we'll use for from-scratch training
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from.startswith('gpt2'):
        if master_process:
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            print(f"Batch size {batch_size} gradient acc steps {gradient_accumulation_steps} batch split size {batch_split_size}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # removed 'resume' and 'gpt2' options

    if master_process:
        print(model.config)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp and actual_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    return model, model_args, optimizer, scaler

model, model_args, optimizer, scaler = init_model(block_size=block_size)

# partition layers across GPUs
num_gpus = torch.cuda.device_count()
layers_to_partition = model.transformer.h # 12 blocks of Self Attention + MLP
# partition_sizes =  [7, 5]
partition_sizes =  [16, 8]
assert sum(partition_sizes) == len(layers_to_partition)
# partition_sizes = [len(layers_to_partition) // num_gpus] * num_gpus
# for i in range(len(layers_to_partition) % num_gpus):
#     partition_sizes[i] += 1

partitions = []
start = 0
for i, size in enumerate(partition_sizes):
    layers = torch.nn.Sequential() 
    # Add the transformer blocks for this partition
    for j in range(start, start + size):
        layers.add_module(f"block_{j}", layers_to_partition[j])
    partitions.append(layers)#.to(f'cuda:{i}'))
    start += size

def get_embedding(model, X):
    _, t = X.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
    tok_emb = model.transformer.wte(X) # token embeddings of shape (b, t, n_embd)
    pos_emb = model.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
    X = model.transformer.drop(tok_emb + pos_emb)
    return X

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.to('cuda:0')
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            #X, Y = get_batch(split)
            with ctx:
                #logits, loss = model(X, Y)
                loss = train_step(data_split=split, to_train=False)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
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

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if abs(num) < 1024.0:
            return f"{num:3.2f}{unit}B"
        num /= 1024.0

def train_step(iter_num=0, data_split='train', to_train=True):
    # determine and set the learning rate for this iteration
    if to_train:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr      

    params_0 = list(partitions[0].parameters()) + \
                list(model.transformer.wte.parameters()) + \
                list(model.transformer.wpe.parameters()) + \
                list(model.transformer.drop.parameters())
    
    params_1 = list(model.lm_head.parameters()) + \
                list(model.transformer.ln_f.parameters()) + \
                list(partitions[1].parameters()) 

    # X, Y = get_batch(data_split)         
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp and actual_ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            X, Y = get_batch(data_split) 
            X_splits = iter(X.split(batch_split_size, dim=0))                    
            Y_splits = iter(Y.split(batch_split_size, dim=0))
            # if ddp_rank == 0:
            #     X_prev = get_embedding(model, next(X_splits))
            #     X_prev = partitions[0](X_prev)
            #     broadcast(X_prev, src=0)

            for X_next in X_splits:
                if ddp_rank == 0:
                    X_next = get_embedding(model, X_next)
                    X_next = partitions[0](X_next)
                    broadcast(X_next, src=0)
            
                if ddp_rank == num_gpus-1:
                    X_prev_intermediate = torch.zeros((batch_split_size, model_args['block_size'], model_args['n_embd'])).to('cuda:1')
                    broadcast(X_prev_intermediate, src=0)
                    
                    X_split = partitions[1].to('cuda:1')(X_prev_intermediate)
                    X_split = model.transformer.ln_f.to(f'cuda:1')(X_split)
                    X_split = model.lm_head.to(f'cuda:1')(X_split)
                    
                    Y_split = next(Y_splits)
                    loss = F.cross_entropy(X_split.view(-1, X_split.size(-1)), Y_split.view(-1).to(f'cuda:1'), ignore_index=-1)
                    loss = loss / (gradient_accumulation_steps*batch_size//batch_split_size) # scale the loss to account for gradient accumulation
                    
                    loss.backward()
                    broadcast(X_prev_intermediate.grad, src=1)
                    for param in params_1:
                        broadcast(param.grad, src=1)
                
                if ddp_rank == 0:
                    for param in params_1:
                        broadcast(param.grad, src=1)
                    broadcast(X_next.grad, src=1)
                    X_next.backward(X_next.grad)

                    for param in params_0:
                        broadcast(param.grad, src=0)
                
                if ddp_rank == 1:
                    for param in params_0:
                        broadcast(param.grad, src=0)
                    loss = torch.Tensor([0]) # dummy value to return for other threads
                
                if to_train:
                    # clip the gradient
                    if grad_clip != 0.0:
                        # scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    # # step the optimizer and scaler if training in fp16
                    # scaler.step(optimizer)
                    # scaler.update()
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)
            
    return loss


if master_process:
    print("\nMeasuring training quality\n")

# training loop
# X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()

with profile(activities=[torch.profiler.ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    while True:
        
        # evaluate the loss on train/val sets and write checkpoints
        # if iter_num % eval_interval == 0:# and master_process:
        #     t0_step = time.time()
        #     losses = estimate_loss()
        #     t1_step = time.time()
        #     if ddp_rank == num_gpus-1:
        #         print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {(t1_step - t0_step)*1000:.2f}ms")
        # if iter_num == 0 and eval_only:
        #     break

        loss = train_step(iter_num=iter_num)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and ddp_rank == num_gpus-1:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * (gradient_accumulation_steps*batch_size//batch_split_size)
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

prof.export_chrome_trace(f"trace{ddp_local_rank}.json")
barrier()

if master_process:
    with open('trace0.json') as f:
        t0 = json.load(f)
    with open('trace1.json') as f:
        t1 = json.load(f)
    del t0['distributedInfo']
    t0['traceEvents'].extend(t1['traceEvents'])
    new_traceEvents = []
    for t in t0['traceEvents']:
        if t['tid'] == 7:
            t['tid'] = t['pid']
            t['pid'] = 0
            new_traceEvents.append(t)
    t0['traceEvents'] = new_traceEvents
    with open('combined_trace.json', 'w+') as f:
        json.dump(t0, f)


print("Peak memory usage for GPUs: ", end="")
for i in range(num_gpus):
    if ddp_rank == i:
        print(
            f"cuda:{i}: {sizeof_fmt(torch.cuda.max_memory_allocated(i))}, ",
            end="",
        )

print()

# -----------------------------------------------------------------------------
# Training throughput
# -----------------------------------------------------------------------------

# function to measure training latency
def measure_training_throughput(batch_size, block_size=128, start_iter=20, max_iters=40, profile_till=None):
    
    # model init
    model, model_args, optimizer, scaler = init_model(block_size=block_size)
    
    if profile_till:
        profiler = profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                           record_shapes=True) 
        profiler.__enter__()

    t0 = 0
    # Do an extra iteration: first iteration is slower as get_batch doesn't overlap with backward pass
    for iter_num in range(max_iters + 1):
        if iter_num == start_iter:
            t0 = time.time()
        loss = train_step(iter_num=iter_num)
        if iter_num == profile_till:
            profiler.__exit__(None,None,None,None)
        if iter_num % 5 == 0 and ddp_rank == num_gpus-1:
            print(f'Iteration {iter_num}: loss {loss.item() * (gradient_accumulation_steps*batch_size//batch_split_size)}')
        
        
    training_total_seconds = time.time() - t0
    tokens_per_iter = ddp_world_size * batch_size * block_size
    training_tokens_per_second = tokens_per_iter * (max_iters - start_iter) / training_total_seconds
    if ddp_rank == num_gpus - 1:
        print(f"Training throughput, batch size {batch_size}: {training_tokens_per_second:.4f} tokens/second")

    print("Peak memory usage for GPUs: ", end="")
    for i in range(num_gpus):
        if ddp_rank == i:
            print(
                f"cuda:{i}: {sizeof_fmt(torch.cuda.max_memory_allocated(i))}, ",
                end="",
            )

    profiler.export_chrome_trace(f"trace{ddp_local_rank}.json")
    barrier()

    if master_process:
        with open('trace0.json') as f:
            t0 = json.load(f)
        with open('trace1.json') as f:
            t1 = json.load(f)
        del t0['distributedInfo']
        t0['traceEvents'].extend(t1['traceEvents'])
        new_traceEvents = []
        for t in t0['traceEvents']:
            if t['tid'] == 7:
                t['tid'] = t['pid']
                t['pid'] = 0
                new_traceEvents.append(t)
        t0['traceEvents'] = new_traceEvents
        with open(f'trace_batch{batch_size}.json', 'w+') as f:
            json.dump(t0, f)


    
# if master_process:
#     print("\nMeasuring training throughput\n")
# batch_size = 4
# measure_training_throughput(batch_size, profile_till=5)
# batch_size = 12
# measure_training_throughput(batch_size)

if ddp:
    destroy_process_group()
