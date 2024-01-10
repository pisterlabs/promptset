import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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
init_from = 'gpt2'#'gpt2-medium'#'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'cs229s'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'shakespeare' #'wikitext'
gradient_accumulation_steps = 4 #5 * 8 # used to simulate larger batch sizes
batch_size = 2 #11 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 5 # total number of training iterations
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
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
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

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
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

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

print("\nMeasuring training quality\n")














####################################################################################################
#####################              PIPELINE PARALLELISM EXTRA CREDIT                     ###########
####################################################################################################
def train_step(iter_num=0, data_split='train', to_train=True):
    # determine and set the learning rate for this iteration
    if to_train:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr      

    # X, Y = get_batch(data_split)         
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp: # and actual_ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            X, Y = get_batch(data_split) 
            X_splits = iter(X.split(batch_split_size, dim=0))                    
            Y_splits = iter(Y.split(batch_split_size, dim=0))
            for X_split in X_splits:
                if ddp_rank == 0:
                    X_split = get_embedding(model, X_split)
                    X_split = partitions[0](X_split)
                    broadcast(X_split, src=0)
            
                if ddp_rank == num_gpus-1:
                    X_split = torch.zeros((batch_split_size, model_args['block_size'], model_args['n_embd'])).to('cuda:1')
                    broadcast(X_split, src=0)
                    
                    X_split = partitions[1].to('cuda:1')(X_split)                    
                    X_split = model.transformer.ln_f.to(f'cuda:1')(X_split)
                    X_split = model.lm_head.to(f'cuda:1')(X_split)
                    
                    Y_split = next(Y_splits)
                    loss = F.cross_entropy(X_split.view(-1, X_split.size(-1)), Y_split.view(-1).to(f'cuda:1'), ignore_index=-1)
                    loss = loss / (gradient_accumulation_steps*batch_size//batch_split_size) # scale the loss to account for gradient accumulation

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        if ddp_rank == num_gpus-1 and to_train:
            scaler.scale(loss).backward()
    if ddp_rank == num_gpus-1:
        if to_train:
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
    else:
        loss = torch.Tensor([0]) # dummy value to return for other threads
    
    return loss

# partition layers across GPUs
num_gpus = torch.cuda.device_count()
layers_to_partition = model.transformer.h # 12 blocks of Self Attention + MLP
# partition_sizes =  [7, 5]
partition_sizes =  [18, 6]
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
####################################################################################################
#####################              PIPELINE PARALLELISM EXTRA CREDIT                     ###########
####################################################################################################








####################################################################################################
#####################              PRUNING CODE                                          ###########
####################################################################################################
def l1_unstructured_prune(module, name, amount):
    if not 0 <= amount <= 1:
        raise ValueError("Pruning amount must be a fraction between 0 and 1.")
    tensor = getattr(module, name)
    num_params_to_prune = round(amount * tensor.nelement())
    if num_params_to_prune != 0:
        l1_norm = torch.abs(tensor.view(-1))
        threshold = torch.topk(l1_norm, num_params_to_prune, largest=False).values[-1]
        mask = torch.gt(l1_norm, threshold).float().view_as(tensor)
    else:
        mask = torch.ones_like(tensor)
    module.register_parameter(name + "_orig", torch.nn.Parameter(tensor.detach()))
    module.register_buffer(name + "_mask", mask)
    def apply_mask_hook(module, inputs):
        orig_tensor = getattr(module, name + "_orig")
        mask = getattr(module, name + "_mask")
        pruned_tensor = orig_tensor * mask
        setattr(module, name, torch.nn.Parameter(pruned_tensor))
    hook = module.register_forward_pre_hook(apply_mask_hook)
    return hook

def l2_structured_attn_pruning(model, name, name2, amount, n_head):
    c_attn_layer = getattr(model, name)
    # print("CPROJ ORIGINAL SHAPE: ", getattr(model, "c_proj").weight.data.shape)
    print("ORIGINAL: ", getattr(model, name).weight.data.shape)

    qkv_weights = c_attn_layer.weight.data
    # print(qkv_weights.shape)
    n_embd = qkv_weights.shape[0]//3
    # print(n_embd)
    q, k, v = qkv_weights.split(n_embd, dim=0)
    # print(q.shape, k.shape, v.shape)

    ql2_norm = torch.norm(q, p=2, dim=1)
    qnum_rows_to_keep = int((1 - amount) * ql2_norm.size(0))
    qnum_rows_to_keep -= qnum_rows_to_keep % n_head
    qrows_to_keep = torch.topk(ql2_norm, qnum_rows_to_keep, largest=True).indices

    kl2_norm = torch.norm(k, p=2, dim=1)
    knum_rows_to_keep = int((1 - amount) * kl2_norm.size(0))
    knum_rows_to_keep -= knum_rows_to_keep % n_head
    krows_to_keep = torch.topk(kl2_norm, knum_rows_to_keep, largest=True).indices

    vl2_norm = torch.norm(v, p=2, dim=1)
    vnum_rows_to_keep = int((1 - amount) * vl2_norm.size(0))
    # print(vnum_rows_to_keep)
    vnum_rows_to_keep -= vnum_rows_to_keep % n_head
    # print("NEW: ", n_head, vnum_rows_to_keep)
    vrows_to_keep = torch.topk(vl2_norm, vnum_rows_to_keep, largest=True).indices

    # print("BIAS: ", c_attn_layer.bias)
    new_current_layer = nn.Linear(c_attn_layer.in_features, qnum_rows_to_keep+knum_rows_to_keep+vnum_rows_to_keep, bias=c_attn_layer.bias is not None)
    new_current_layer.weight.data = torch.cat([q[qrows_to_keep, :] , k[krows_to_keep, :] , v[vrows_to_keep, :] ], dim=0)

    setattr(model, name, new_current_layer)
    
    print("PRUNED: ", getattr(model, name).weight.data.shape)

    c_proj_layer = getattr(model, name2)

    new_c_proj_layer = nn.Linear(vnum_rows_to_keep, c_proj_layer.out_features, bias=c_proj_layer.bias is not None)

    new_c_proj_layer.weight.data = c_proj_layer.weight.data[:, vrows_to_keep]
    if c_proj_layer.bias is not None:
        new_c_proj_layer.bias.data = c_proj_layer.bias.data[vrows_to_keep]

    setattr(model, name2, new_c_proj_layer)
    
    # print(getattr(model, name2).in_features, getattr(model, name2).out_features, getattr(model, name2).weight.data.shape)


def apply_unstructured_pruning_to_causal_self_attention(module, pruning_percentage=0.2):
    l1_unstructured_prune(module.c_attn, name='weight', amount=pruning_percentage)

def apply_structured_pruning_to_causal_self_attention(block, pruning_percentage=0.2):
    l2_structured_attn_pruning(block, 'c_attn', 'c_proj', pruning_percentage, n_head)

def prune_gpt_model(model, pruning_percentage=0.2):
    for block in model.transformer.h:
        apply_unstructured_pruning_to_causal_self_attention(block.attn, pruning_percentage)
        # apply_structured_pruning_to_causal_self_attention(block.attn, pruning_percentage)

def calculate_reduction_percentage(current_percentage, target_percentage):
    reduction_factor = target_percentage / current_percentage
    reduction_percentage = (1 - reduction_factor) * 100
    return reduction_percentage

def calculate_percentage_to_reduce_each_step(start_percentage, step_reduction, steps):
    percentages_to_reduce = []
    current_percentage = start_percentage
    for _ in range(steps):
        target_percentage = current_percentage - step_reduction
        reduction_percentage = calculate_reduction_percentage(current_percentage, target_percentage)
        percentages_to_reduce.append(reduction_percentage)
        current_percentage = target_percentage
    return percentages_to_reduce

percentages_to_reduce_each_step = calculate_percentage_to_reduce_each_step(100, 10, 10)  # 10 steps

def inferMe(model):
    init_from = 'gpt2' 
    out_dir = 'out'
    start = "\n" 
    num_samples = 1
    max_new_tokens = 500
    temperature = 0.8 
    top_k = 200 
    seed = 1337
    device = 'cuda' 
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
    compile = False 
    exec(open('configurator.py').read())

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)

    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        # print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)

    def measure_inference_latency(batch_size):
        x = torch.tensor(start_ids * batch_size, dtype=torch.long, device=device).reshape(batch_size, -1)
        with torch.no_grad():
            with ctx:
                tokens_decoded = 0
                inference_start = time.time()
                for k in range(num_samples):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    tokens_decoded += y.shape[0] * (y.shape[1] - 1)
                inference_finish = time.time()
        inference_total_seconds = inference_finish - inference_start
        inference_tokens_per_second = tokens_decoded / inference_total_seconds
        print(f"Inference latency, batch size {batch_size}: {inference_tokens_per_second:.4f} tokens/second")

    # print("\nMeasuring inference latency\n")

    batch_size = 1
    measure_inference_latency(1)
    batch_size = 12
    measure_inference_latency(12)
####################################################################################################
#####################              PRUNING CODE                                          ###########
####################################################################################################









# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    
    # ###########################################################
    ####################### PRUNING CODE                 ########
    # ###########################################################
    # if iter_num>=100 and iter_num%100==0: 
    #     model.eval()
    #     inferMe(model)
    #     model.train()
    # ###########################################################
    # ###########################################################

    # ###########################################################
    ####################### PRUNING CODE                 ########
    # ###########################################################
    # STRUCTURED
    # if iter_num>=100 and iter_num%100==0:
    #     prune_gpt_model(model, pruning_percentage=percentages_to_reduce_each_step[int((iter_num//100) - 1)]/100)
    #     print("Iteration", iter_num, "PRUNED")

    # #UNSTRUCTURED
    # if iter_num>=100 and iter_num%100==0:
    #     prune_gpt_model(model, pruning_percentage=iter_num/1000)
    #     print("Iteration", iter_num, "PRUNED",iter_num/1000, "Percent of Graph")
    # ###########################################################
    # ###########################################################

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

        # ###########################################################
        ####################### PRUNING CODE                 ########
        # ###########################################################
        for name, param in model.named_parameters():
            mask_name = name + '_mask'
            if mask_name in model.state_dict() and param.grad is not None:
                mask = model.state_dict()[mask_name]
                param.grad.data *= mask
        #######################################################
        #######################################################
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    # ###########################################################
    ####################### PRUNING CODE                 ########
    # ###########################################################
    if iter_num>=50 and lossf<=3.2 and steps_since_prune>=25:
        prune_amount = prune_amount + 0.1
        prune_gpt_model(model, pruning_percentage=prune_amount)
        print("Iteration", iter_num, "PRUNED",prune_amount*100, "Percent of Graph")
        steps_since_prune = 0
    steps_since_prune += 1
    #######################################################
    #######################################################
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

print()

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)

# function to measure inference latency
def measure_inference_latency(batch_size):
    x = torch.tensor(start_ids * batch_size, dtype=torch.long, device=device).reshape(batch_size, -1)
    with torch.no_grad():
        with ctx:
            tokens_decoded = 0
            inference_start = time.time()
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                tokens_decoded += y.shape[0] * (y.shape[1] - 1)
            inference_finish = time.time()
    inference_total_seconds = inference_finish - inference_start
    inference_tokens_per_second = tokens_decoded / inference_total_seconds
    print(f"Inference latency, batch size {batch_size}: {inference_tokens_per_second:.4f} tokens/second")

print("\nMeasuring inference latency\n")

batch_size = 1
measure_inference_latency(1)
batch_size = 12
measure_inference_latency(12)
print()
del model

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# function to measure training latency
def measure_training_throughput(batch_size, max_iters=5):
    times = []
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    # Do an extra iteration: first iteration is slower as get_batch doesn't overlap with backward pass
    for _ in range(max_iters + 1):
        with ctx:
            logits, loss = model(X, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        # timing and logging
        t1 = time.time()
        times.append(t1 - t0)
        t0 = t1
    training_total_seconds = sum(times[1:])
    tokens_per_iter = ddp_world_size * batch_size * block_size
    training_tokens_per_second = tokens_per_iter / training_total_seconds
    print(f"Training throughput, batch size {batch_size}: {training_tokens_per_second:.4f} tokens/second")
    
print("\nMeasuring training throughput\n")

batch_size = 4
measure_training_throughput(batch_size)
batch_size = 12
measure_training_throughput(batch_size)

if ddp:
    destroy_process_group()



######################################################################################################
###########################                     MEMORY USAGE FOR LEADERBOARD              ############
######################################################################################################


"""
Evaluate next-token-prediction perplexity of pre-trained model
"""
 
import gc
from model_speculative import GPT
from model_quantized import GPT_Q
from contextlib import nullcontext
import numpy as np
import os
import tiktoken
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
 
# -----------------------------------------------------------------------------
init_from = 'gpt2' # a gpt2 variant (e.g. 'gpt2-xl')
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
dataset = 'wikitext' # 'shakespeare' or 'wikitext'
block_size = 1024
num_warmup = 1 # how many warmups to do before benchmarking
max_new_tokens = 50 # number of tokens generated in each sample
temperature = 0.4 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
speculative_tokens = 3 # how many tokens should the draft model decode?
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
# -----------------------------------------------------------------------------
 
# Setup environment
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
 
# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
 
# Load dataset
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
 
def measure_perplexity(model, data, batch_size):
    nll_weights = []
    nlls = []
    for i in range(0, len(data), block_size * batch_size):
        j = min(i + block_size * batch_size, len(data))
        ix = torch.arange(i, j, block_size)
        x = []
        for k in ix:
            x.append(torch.from_numpy((data[k:k+block_size]).astype(np.int64)))
        x = torch.stack([F.pad(y, (0, block_size - len(y)), value=-1) for y in x])
        nll_weights.append((x != -1).sum().item() / len(data))
        if device_type == 'cuda':
            # pin array x which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        with torch.no_grad():
            with ctx:
                # y = x[:, 1:].clone()
                # x[x == -1] = 0
                # logits, loss = model(x[:, :-1], y)
                # nlls.append(loss)
                # https://github.com/huggingface/transformers/blob/df5c5c62ae253055336f5bb0828ca8e3e15ab6bd/src/transformers/models/gpt2/modeling_gpt2.py#L1099
                y = x.clone()
                x[x == -1] = 0
                logits, _ = model(x, y)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = y[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
                nlls.append(loss)
    nlls = [nll_weights[i] * nlls[i] for i in range(len(nlls))]
    return torch.exp(torch.stack(nlls).sum()).item()
 
# ------------------------------------------------------------------------------
# Leaderboard
# ------------------------------------------------------------------------------
print("-" * 80 + "\nLeaderboard: memory usage for inference without quantization \n" + "-" * 80)
# Load pre-trained model
model_pt = GPT.from_pretrained(init_from, dict(dropout=0.0))
model_pt.eval()
torch.cuda.reset_peak_memory_stats(device=device)
model_pt.to(device)
print(f"\nGPU memory allocated after calling model.to(device) {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    model_pt = torch.compile(model_pt) # requires PyTorch 2.0 (optional)
# Run evaluation
torch.cuda.reset_peak_memory_stats(device=device)
ppl_pt_bs4 = measure_perplexity(model_pt, val_data, batch_size=1)
print(f"GPT-2 perplexity on {dataset}/val.bin, batch_size={1}: {ppl_pt_bs4:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
del model_pt
gc.collect()
torch.cuda.empty_cache()
print("\n")
 
print("-" * 80 + "\nLeaderboard: memory usage for inference with quantization\n" + "-" * 80)
# Load pre-trained model and quantize
model_dq = GPT_Q.from_pretrained(init_from, dict(dropout=0.0))
model_dq.quantize_all_parameters()
torch.cuda.reset_peak_memory_stats(device=device)
model_dq.to(device)
print(f"\nGPU memory allocated after calling model.to(device) {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    model_dq = torch.compile(model_dq) # requires PyTorch 2.0 (optional)
# Run evaluation
torch.cuda.reset_peak_memory_stats(device=device)
ppl_dq_bs4 = measure_perplexity(model_dq, val_data, batch_size=1)
print(f"GPT-2 quantized perplexity on {dataset}/val.bin, batch_size={1}: {ppl_dq_bs4:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
print()
del model_dq
gc.collect()
torch.cuda.empty_cache()
 



