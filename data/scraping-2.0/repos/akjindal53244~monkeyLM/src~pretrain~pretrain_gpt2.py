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
from typing import Optional, Any, Type, Literal

import numpy as np
import torch

from src.gpt2.gpt2_model import GPTConfig, GPT
from src.common.utils.compute_utils import get_tokens_per_iter, get_device_type, get_pt_dtype, get_ctx
from src.common.constants import gpu_promised_tflops_map, my_gpu_name

model_args = {}

def get_batch(split_name, train_data, val_data):
    data = train_data if split_name == 'train' else val_data
    block_size, batch_size = model_args["block_size"], model_args["batch_size"]
    device, device_type = model_args["device"], model_args["device_type"]

    # randomly choose [[batch_size]] token indices in training data.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking = True), y.pin_memory().to(device, non_blocking = True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    eval_iters = model_args["eval_iters"]
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            with get_ctx(model_args["device"], model_args["dtype"]):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    warmup_iters = model_args["warmup_iters"]
    learning_rate = model_args["learning_rate"]
    lr_decay_iters = model_args["lr_decay_iters"]
    min_lr = model_args["min_lr"]

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


def setup(
    dataset: str = 'openwebtext',
    block_size: int = 1024,
    batch_size: int = 12,  # if gradient_accumulation_steps > 1, this is the micro-batch size
    gradient_accumulation_steps: int = 5 * 8,  # used to simulate larger batch sizes
    vocab_size = None,

    max_iters: int = 50000,  # total number of training iterations

    n_embd: int = 768,
    n_head: int = 12,
    n_layer: int = 12,
    bias: bool = False,

    # LR
    learning_rate: float = 6e-4,  # max learning rate
    decay_lr: bool = True,  # whether to decay the learning rate
    warmup_iters: int = 5,  # how many steps to warm up for
    lr_decay_iters: int = 600000,  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5,  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # optimizer
    optimizer: str = 'custom',

    # regularization
    dropout: float = 0.0, # for pretraining 0 is good, for finetuning try 0.1+

    weight_decay: float = 1e-1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    grad_clip: float = 1.0,  # clip gradients at this value, or disable if == 0.0

    # normalization layers
    norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm",
    norm_eps: float = 1e-5,

    # Eval settings
    eval_iters: int = 10,
    eval_interval: int = 10,

    # inference mode
    eval_only: bool = False, # if True, script exits right after the first eval

    # logging setting
    log_interval: int = 1,

    # wandb logging
    wandb_log: bool = False,  # disabled by default
    wandb_project: str = 'owt',
    wandb_run_name: str = 'gpt2',  # 'run' + str(time.time())

    # model output
    out_dir: str = 'out',
    always_save_checkpoint: bool = True,  # if True, always save a checkpoint after each eval

    # model initialization. 'scratch': pretrain from scratch. 'gpt2' means load pretrained weights from huggingface.
    # 'resume': further pretraining
    init_from: str = 'scratch',

    # GPU settings
    device: str = 'cuda',  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    compile: bool = True,  # use PyTorch 2.0 to compile the model to be faster
    # seed
    seed: int = 1337,
):
    global model_args
    model_args = {"dataset"                    : dataset,
                  "block_size"                 : block_size,
                  "batch_size"                 : batch_size,
                  "gradient_accumulation_steps": gradient_accumulation_steps,
                  "vocab_size"                 : vocab_size,
                  "max_iters"                  : max_iters,
                  "n_embd"                     : n_embd,
                  "n_head"                     : n_head,
                  "n_layer"                    : n_layer,
                  "bias"                       : bias,
                  "learning_rate"              : learning_rate,
                  "decay_lr"                   : decay_lr,
                  "warmup_iters"               : warmup_iters,
                  "lr_decay_iters"             : lr_decay_iters,
                  "min_lr"                     : min_lr,
                  "optimizer"                  : optimizer,
                  "dropout"                    : dropout,
                  "weight_decay"               : weight_decay,
                  "beta1"                      : beta1,
                  "beta2"                      : beta2,
                  "grad_clip"                  : grad_clip,
                  "norm_class"                 : norm_class,
                  "norm_eps"                   : norm_eps,
                  "eval_iters"                 : eval_iters,
                  "eval_interval"              : eval_interval,
                  "eval_only"                  : eval_only,
                  "log_interval"               : log_interval,
                  "wandb_log"                  : wandb_log,
                  "wandb_project"              : wandb_project,
                  "wandb_run_name"             : wandb_run_name,
                  "out_dir"                    : out_dir,
                  "always_save_checkpoint"     : always_save_checkpoint,
                  "init_from"                  : init_from,
                  "device"                     : device,
                  "dtype"                      : dtype,
                  "compile"                    : compile,
                  "seed"                       : seed,
                  "device_type"                : get_device_type(device),
                  "pt_dtype"                   : get_pt_dtype(dtype)
    }

    tokens_per_iter = get_tokens_per_iter(block_size, batch_size, gradient_accumulation_steps)
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    # allow tf32 on matmul on new NVIDIA GPUs since Ampere. Defaults to False for pytorch 1.12 and later.
    # tf32 stands for TensorFloat32 tensor cores
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # poor man's data loader
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        model_args["vocab_size"] = 50304  # nearest multiple of 64 for efficiency purpose

        gptconf = GPTConfig(
            block_size = model_args["block_size"],
            vocab_size = model_args["vocab_size"],
            n_layer = model_args["n_layer"],
            n_head = model_args["n_head"],
            n_embd = model_args["n_embd"],
            dropout = model_args["dropout"],
            bias = model_args["bias"]
        )
        model = GPT(gptconf)
    elif init_from == 'resume':
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        print(f"Resuming training from {ckpt_path}..")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        print(f"Overriding model architecture parameters: n_layer, n_head, n_embd, block_size, bias, and vocab_size from {ckpt_path}..")
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(
            block_size = model_args["block_size"],
            vocab_size = model_args["vocab_size"],
            n_layer = model_args["n_layer"],
            n_head = model_args["n_head"],
            n_embd = model_args["n_embd"],
            dropout = model_args["dropout"],
            bias = model_args["bias"]
        )
        model = GPT(gptconf)
        state_dict = checkpoint['model']

        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.' # it is added as a result of model.compile()
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        print("Loading model state from Checkpoint..")
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI {init_from} weights from Huggingface..")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        print("Overriding model architecture parameters: n_layer, n_head, n_embd, block_size, bias,vocab_size from model checkpoint..")
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        print(f"block_size {block_size} is less than model's block_size {model.config.block_size} => cropping model block size..")
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    print(f"Moving model to device: {model_args['device']}")
    model.to(model_args["device"])

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled = (dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(
        model_args["weight_decay"],
        model_args["learning_rate"],
        (model_args["beta1"], model_args["beta2"]),
        model_args["device_type"]
    )
    if init_from == 'resume':
        print("Loading Optimizer State from Checkpoint..")
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model) # requires PyTorch 2.0

    # logging
    if model_args["wandb_log"]:
        import wandb
        wandb.init(project=model_args["wandb_project"], name=model_args["wandb_run_name"], config=model_args)

    # training loop
    X, Y = get_batch('train', train_data, val_data) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if model_args["decay_lr"] else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % model_args["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if model_args["wandb_log"]:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            if losses['val'] < best_val_loss or model_args["always_save_checkpoint"]:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"saving checkpoint to {model_args['out_dir']}")
                    torch.save(checkpoint, os.path.join(model_args['out_dir'], 'ckpt.pt'))
        if iter_num == 0 and model_args["eval_only"]:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(model_args["gradient_accumulation_steps"]):
            with get_ctx(model_args["device"], model_args["dtype"]):
                logits, loss = model(X, Y)
                loss = loss / model_args["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train', train_data, val_data)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if model_args["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_args["grad_clip"])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % model_args["log_interval"] == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * model_args["gradient_accumulation_steps"]
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = model.estimate_mfu(model_args["batch_size"] * model_args["gradient_accumulation_steps"], dt, gpu_promised_tflops_map[my_gpu_name])
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > model_args["max_iters"]:
            break

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
