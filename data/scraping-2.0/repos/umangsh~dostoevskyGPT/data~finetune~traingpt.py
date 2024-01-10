"""Finetune GPT-2 model on Dostoevsky's works."""

import time

import numpy as np
import torch
from numpy.typing import NDArray

from common import nanogpt


def get_batch(  # pylint: disable=too-many-arguments
    split: str, train_data: NDArray, val_data: NDArray, block_size: int, batch_size: int, device: str
) -> tuple:
    """Get a batch of data from either the train or validation split."""
    data = train_data if split == "train" else val_data
    index = torch.randint(len(data) - block_size, (batch_size,))
    batch_x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in index])
    batch_y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in index])
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    return batch_x, batch_y


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(  # pylint: disable=too-many-arguments, too-many-locals
    train_data: NDArray,
    val_data: NDArray,
    block_size: int,
    batch_size: int,
    device: str,
    model: nanogpt.GPT,
    ctx: torch.amp.autocast,
    eval_iters: int,
) -> dict:
    """Estimate loss over either the train or validation split."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for iteration in range(eval_iters):
            batch_x, batch_y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            with ctx:
                _logits, loss = model(batch_x, batch_y)
            losses[iteration] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(  # pylint: disable=too-many-locals, too-many-statements
    checkpoint_file: str, train_file: str, validation_file: str
) -> None:
    """Finetune GPT-2 model on Dostoevsky's works."""
    # default config values
    eval_interval = 5
    log_interval = 1
    eval_iters = 40
    init_from = "gpt2"

    # data
    gradient_accumulation_steps = 32  # used to simulate larger batch sizes
    batch_size = 1
    block_size = 1024

    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0
    bias = True

    # adamw optimizer
    learning_rate = 3e-5  # max learning rate
    max_iters = 20  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

    # system
    device = "mps"

    time_seed = int(time.time())
    torch.manual_seed(time_seed)
    ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)

    # data loader
    train_data = np.memmap(train_file, dtype=np.uint16, mode="r")
    val_data = np.memmap(validation_file, dtype=np.uint16, mode="r")

    # model init
    model_args = {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
        "bias": bias,
        "vocab_size": None,
        "dropout": dropout,
    }

    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")  # noqa: T201

    # initialize from OpenAI GPT-2 weights
    override_args = {"dropout": dropout}
    model = nanogpt.GPT.from_pretrained(init_from, override_args)

    # read off the created config params, so we can store them into checkpoint correctly
    for arg in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[arg] = getattr(model.config, arg)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = block_size  # so that the checkpoint will have the right value
    model.to(device)

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

    # training loop
    iter_num = 0
    best_val_loss = 1e9
    batch_x, batch_y = get_batch(
        "train", train_data, val_data, block_size, batch_size, device
    )  # fetch the very first batch
    start_time = time.time()
    while True:
        # termination conditions
        if iter_num > max_iters:
            break

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss(train_data, val_data, block_size, batch_size, device, model, ctx, eval_iters)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")  # noqa: T201
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                print(f"saving checkpoint to {checkpoint_file}")  # noqa: T201
                torch.save(checkpoint, checkpoint_file)

        # forward backward update
        for _unused in range(gradient_accumulation_steps):
            with ctx:
                _logits, loss = model(batch_x, batch_y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            batch_x, batch_y = get_batch("train", train_data, val_data, block_size, batch_size, device)

        # clip the gradient
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        current_time = time.time()
        time_delta = current_time - start_time
        start_time = current_time
        if iter_num % log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            print(f"iter {iter_num}: loss {lossf:.4f}, time {time_delta*1000:.2f}ms")  # noqa: T201
        iter_num += 1


def main() -> None:
    """Main method."""
    # Initialize the training arguments.
    checkpoint_file = "/Users/umang/Desktop/github/dostoevskyGPT/out/finetune/checkpoint.pt"
    train_file = "/Users/umang/Desktop/github/dostoevskyGPT/data/finetune/train.bin"
    validation_file = "/Users/umang/Desktop/github/dostoevskyGPT/data/finetune/val.bin"

    train(checkpoint_file, train_file, validation_file)


if __name__ == "__main__":
    main()
