from collections import defaultdict
import os
from pathlib import Path

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from BLIP2.constant import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from dataset import IterLoader, load_dataset, ImageDirDataset
from lavis.processors.blip_processors import BlipCaptionProcessor
from modeling_zerobooth_finetune import ZeroBooth
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

from torch.utils.data import DistributedSampler

from diffusers.optimization import get_scheduler

logger = get_logger(__name__)

use_cache_subj_emb = True
train_text_encoder = False

model_config = {
    "train_text_encoder": train_text_encoder,
    # "train_unet": False,
    "train_unet": "crossattn-kv", # crossattn-kv: only tune KV, upblocks: freeze all downblocks and middleblocks
    # STABLE DIFFUSION
    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "revision": None,
    # BLIP
    "text_model": "bert-base-uncased",
    "pretrained": "/export/share/junnan-li/BLIP2/checkpoint/clip_q16.pth",
    "vision_model": "clip",
    "image_size": 224,
    "num_query_token": 16,
    "max_text_length": 32,
    "embed_dim": 256,
    "use_grad_checkpointing": True
}
# model_config = SimpleNamespace(**model_config)

def create_transforms(image_size=224, tgt_image_size=512):
    # preprocess
    # blip image transform
    inp_image_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )

    # stable diffusion image transform
    tgt_image_transform = transforms.Compose(
        [
            transforms.Resize(
                tgt_image_size, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(tgt_image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    text_transform = BlipCaptionProcessor()

    return {
        "inp_image_transform": inp_image_transform,
        "tgt_image_transform": tgt_image_transform,
        "text_transform": text_transform,
    }


def unwrap_dist_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model

def move_batch_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


def collect_subj_inputs(dataset, device):
    examples = [dataset[i] for i in range(dataset.actual_len)]

    input_images_key = "input_image"
    input_images = (
        torch.stack([example[input_images_key] for example in examples])
        .to(memory_format=torch.contiguous_format)
        .float()
    )

    class_names = [example["class_name"] for example in examples]

    return {
        "input_images": input_images.to(device),
        "class_names": class_names,
    }


def main(
    subject,
    image_dir,
    output_dir,
    logging_dir,
    checkpoint,
    learning_rate,
    train_batch_size,
    max_train_steps,
    lr_scheduler="constant",
    lr_warmup_steps=0,
    max_grad_norm=20.0,
    seed=1337,
    save_model=True,
    save_steps=50,
    min_save_steps=50,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=0.01,
    adam_epsilon=1e-8,
    mixed_precision="no"
):

    logging_dir = Path(output_dir, logging_dir)

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        # mixed_precision=mixed_precision,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    # only process rank 0 should log
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if accelerator.is_main_process or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    # =============== model ==================
    processors = create_transforms(224, 512)
    model = ZeroBooth(config=model_config)

    # load checkpoint
    print("loading checkpoint: ", checkpoint)
    model.load_checkpoint(checkpoint)

    # ====== Dataset ======
    def collate_fn(examples):
        # random choice from a "referring" batch and a "completion" batch
        # is_referring = random.choice([True, False])
        input_images_key = "input_image"
        pixel_values_key = "target_image"
        input_id_key = "input_ids_label"

        ctx_begin_pos_key = "ctx_begin_pos_label"

        batch_type = "one-stage"

        input_ids = [example[input_id_key] for example in examples]
        ctx_begin_pos = [example[ctx_begin_pos_key] for example in examples]
        pixel_values = (
            torch.stack([example[pixel_values_key] for example in examples])
            .to(memory_format=torch.contiguous_format)
            .float()
        )
        input_images = (
            torch.stack([example[input_images_key] for example in examples])
            .to(memory_format=torch.contiguous_format)
            .float()
        )

        input_ids = (
            unwrap_dist_model(model)
            .tokenizer.pad(
                {"input_ids": input_ids},
                padding="longest",
                # max_length=tokenizer.model_max_length,
                max_length=35,
                return_tensors="pt",
            )
            .input_ids
        )
        class_names = [example["class_name"] for example in examples]

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "input_images": input_images,
            "class_names": class_names,
            "ctx_begin_pos": ctx_begin_pos,
            "batch_type": batch_type,
        }

        return batch

    print("Loading dataset")

    train_dataset = load_dataset(
        dataset_name="imagedir",
        inp_image_transform=processors["inp_image_transform"],
        tgt_image_transform=processors["tgt_image_transform"],
        inp_bbox_transform=None,
        tgt_bbox_transform=None,
        msk_bbox_transform=None,
        # inp_bbox_transform=processors["inp_bbox_transform"],
        # tgt_bbox_transform=processors["tgt_bbox_transform"],
        text_transform=processors["text_transform"],
        clip_tokenizer=model.tokenizer,
        subject=subject,
        image_dir=image_dir,
        shuffle_input=False,
    )
    print(f"Loaded {len(train_dataset)} training examples")

    if use_cache_subj_emb:
        model = model.cuda()

        subj_inputs = collect_subj_inputs(
            train_dataset,
            device=accelerator.device
        )
        model.init_ctx_embeddings_cache(subj_inputs)
        # to save memory
        model.move_ctx_encoder_to_cpu()
        torch.cuda.empty_cache()

    for x in model.named_parameters():
        print(x[0], x[1].requires_grad)

    # optimization
    optimizer_class = torch.optim.AdamW
    model_params = model.parameters()

    optimizer = optimizer_class(
        model_params,
        lr=float(learning_rate),
        betas=(float(adam_beta1), float(adam_beta2)),
        weight_decay=float(adam_weight_decay),
        eps=float(adam_epsilon),
    )

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=sampler,
    )

    train_dataloader = IterLoader(train_dataloader)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps
    )

    (
        model,
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        lr_scheduler,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers("dreambooth", config=vars(args))
        accelerator.init_trackers("dreambooth")

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    save_to = os.path.join(output_dir, f"{global_step}")

    # for epoch in range(args.num_train_epochs):
    model.train()

    # for step, batch in enumerate(train_dataloader):
    while True:
        batch = next(train_dataloader)
        move_batch_to_device(batch, accelerator.device)

        loss = model(batch)

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model_params, max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        # Create the pipeline using using the trained modules and save it.
        if global_step % save_steps == 0 or global_step == max_train_steps:
            if global_step > min_save_steps:
                if save_model:
                    print(f"Saving model at step {global_step}.")
                    save_to = os.path.join(output_dir, f"{global_step}")

                    if accelerator.is_main_process:
                        unwrap_dist_model(model).save_checkpoint(save_to, accelerator)

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()

    model.cpu()
    del model
    torch.cuda.empty_cache()


def generate_annotations():
    annotation_path = os.path.join(image_dir, "annotations.json")

    if force_init_annotations or not os.path.exists(annotation_path):
        print("Generating annotations...")
        ImageDirDataset.generate_annotations(
            subject=subject,
            image_dir=image_dir,
            annotation_path=annotation_path,
        )
    torch.cuda.empty_cache()

if __name__ == "__main__":
    debug = False

    train_batch_size = 3
    learning_rate = 5e-6
    # max_train_steps = 300
    # max_train_steps = 130
    min_save_steps = 30
    save_step = 10

    force_init_annotations = False

    data_root = "/export/home/workspace/dreambooth/diffusers/official_benchmark/dreambooth/dataset"
    instances = [
        # {"subject": "backpack", "image_dir": "backpack"},
        # {"subject": "backpack", "image_dir": "backpack_dog"},
        # {"subject": "plushie",  "image_dir": "bear_plushie"},
        # {"subject": "bowl", "image_dir": "berry_bowl"},
        # {"subject": "can", "image_dir": "can"},
        # {"subject": "candle", "image_dir": "candle"},
        # {"subject": "cat", "image_dir": "cat"},
        # {"subject": "cat", "image_dir": "cat2"},
        # {"subject": "clock", "image_dir": "clock"},
        # {"subject": "sneaker", "image_dir": "colorful_sneaker"},
        # {"subject": "dog", "image_dir": "dog"},
        # {"subject": "dog", "image_dir": "dog2"},
        # {"subject": "dog", "image_dir": "dog3"},
        # {"subject": "dog", "image_dir": "dog5"},
        # {"subject": "dog", "image_dir": "dog6"},
        # {"subject": "dog", "image_dir": "dog7"},
        # {"subject": "dog", "image_dir": "dog8"},
        # {"subject": "toy", "image_dir": "duck_toy"},
        # {"subject": "boot", "image_dir": "fancy_boot"},
        # {"subject": "plushie", "image_dir": "grey_sloth_plushie"},
        # {"subject": "toy", "image_dir": "monster_toy"},
        # {"subject": "sunglasses", "image_dir": "pink_sunglasses"},
        {"subject": "toy", "image_dir": "poop_emoji"},
        # {"subject": "car", "image_dir": "rc_car"},
        # {"subject": "cartoon", "image_dir": "red_cartoon"},
        # {"subject": "robot", "image_dir": "robot_toy"},
        # {"subject": "sneaker", "image_dir": "shiny_sneaker"},
        # {"subject": "teapot", "image_dir": "teapot"},
        # {"subject": "bottle", "image_dir": "vase"},
        # {"subject": "plushie", "image_dir": "wolf_plushie"},
    ]

    subj2steps = {"cat": 70, "dog": 70, "bowl": 50}
    subj2steps = defaultdict(lambda: 120, subj2steps)

    subj2min_steps = {"cat": 10, "dog": 10, "bowl": 0}
    subj2min_steps = defaultdict(lambda: 40, subj2min_steps)

    # default args
    checkpoint = "/export/home/workspace/dreambooth/diffusers/output/pretrain-202302315-unet-textenc-v1.5-capfilt6b7-synbbox-matting-rr0-drop15-500k/500000"

    import datetime
    from datetime import datetime

    for instance in instances:
        subject = instance["subject"]
        image_dir = os.path.join(data_root, instance["image_dir"])

        max_train_steps = subj2steps[subject]
        min_save_steps = subj2min_steps[subject]

        generate_annotations()

        # get now
        image_dir_base = os.path.basename(image_dir)

        output_dir = "/export/home/workspace/dreambooth/diffusers/output/benchmark/{}-{}-{}-{}-cache={}-textenc={}".format(
            datetime.now().strftime("%y%m%d%H%M%S"), image_dir_base, max_train_steps, learning_rate, use_cache_subj_emb, train_text_encoder,
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logging_dir = os.path.join(output_dir, "logs")

        main(
            subject,
            image_dir,
            output_dir,
            logging_dir,
            checkpoint,
            learning_rate,
            train_batch_size,
            max_train_steps,
            save_steps=save_step,
            min_save_steps=min_save_steps,
        )

        torch.cuda.empty_cache()
        print("Done. Checkpoint saved to: {}".format(output_dir))
