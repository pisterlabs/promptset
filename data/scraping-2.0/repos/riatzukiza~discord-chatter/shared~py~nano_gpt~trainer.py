import torch
import time
import math
import os
import numpy as np
import tiktoken

from contextlib import nullcontext
from shared.nano_gpt.model import GPTConfig, GPT

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
master_process = True
seed_offset = 0
ddp_world_size = 1
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
def setup_model(
        out_dir,
        init_from = 'scratch', # 'scratch' or 'resume' or 'gpt2*'
        block_size=1024,
        device = 'cuda', # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        n_layer=12,
        n_head=12,
        n_embd=768,
        bias=False,
        dropout=0.2,
        weight_decay=1e-1,
        learning_rate=6e-4,
        beta1=0.9,
        beta2=0.95,
):

    os.makedirs(out_dir, exist_ok=True)
    iter_num=0
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    best_val_loss = 1e9
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    if init_from == 'scratch':
        model=init_from_scratch(model_args)
    elif init_from == 'resume':
        model,checkpoint=resume_model(out_dir, model_args,device=device)
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
    else:
        raise Exception("Improperly formatted init_from")
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)

    # compile the model
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0


    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    return model, model_args, iter_num, best_val_loss, checkpoint, scaler, optimizer

def encode_document(doc:str,encoder):
    ids = encoder.encode_ordinary(doc)
    ids.append(encoder.eot_token)
    return ids

# def get_document_by_id(id,encoded_ids, document_ids):
#     document=



def encode_training_documents(documents:list,encoder):
    data = []
    document_ids=[]
    current_document_id=0
    for document in documents:
        ids = encode_document(document,encoder)
        # current_document_id= len(ids)+current_document_id
        # document_ids.push(document_id)
        data=data+ids
    return data,document_ids


def split_documents_for_evaluation(documents, split,encoder):
    document_data,document_ids=encode_training_documents(documents,encoder)
    n=len(document_data)

    training_data = np.array(document_data[:int(n*split)],dtype=np.uint16)
    evaluation_data = np.array(document_data[int(n*split):],dtype=np.uint16)
    return training_data,evaluation_data

    # # encode with tiktoken gpt2 bpe
    # enc = tiktoken.get_encoding("gpt2")
    # train_ids = enc.encode_ordinary(train_data)
    # val_ids = enc.encode_ordinary(val_data)

    # print(f"train has {len(train_ids):,} tokens")
    # print(f"val has {len(val_ids):,} tokens")

    # # export to bin files
    # train_ids = np.array(train_ids, dtype=np.uint16)
    # val_ids = np.array(val_ids, dtype=np.uint16)

    # return train_ids, val_ids


def init_from_scratch(model_args,meta_vocab_size=50304):
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None: print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    return GPT(gptconf)

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
def resume_model(out_dir,model_args,device):
    print(f"Resuming from {out_dir} device {device}")
    # resume training from a checkpoint.
    try:
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
    except:
        ckpt_path = os.path.join(out_dir, 'ckpt.pt.backup')
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
    return model, checkpoint

def train_gpt_model(
        model, iter_num, best_val_loss, checkpoint, scaler, optimizer,model_args,
        out_dir,
        input_data,
        gradient_accumulation_steps=5 * 8,
        batch_size=8,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        bias=False,
        device="cuda",
        learning_rate=6e-4,
        warmup_iters=2000,
        eval_iters=200,
        lr_decay_iters=600000,
        min_lr=6e-5,
        decay_lr=True,
        eval_interval=1,
        always_save_checkpoint=False,
        eval_only=False,
        grad_clip=1.0,
        log_interval = 1,
        max_iters = 600000
):

    config={
        'n_layer':n_layer,
        'n_head':n_head,
        'n_embd':n_embd,
        'block_size':block_size,
        'bias':bias,
        'vocab_size':None
    }
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")


    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    train_data, val_data=split_documents_for_evaluation(input_data, 0.9, tiktoken.get_encoding("gpt2"))

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

    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model =  model
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, best val loss ({best_val_loss})")
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
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt.backup'))
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
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
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"local_iter {local_iter_num} total_iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if local_iter_num > max_iters:
            break
    return model, model_args, iter_num, best_val_loss, checkpoint, scaler,optimizer
