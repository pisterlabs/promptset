import argparse
import itertools
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import torch
import torch.utils.checkpoint
import yaml
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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def create_transforms(config):
    # preprocess
    # blip image transform
    inp_image_transform = transforms.Compose(
        [
            transforms.Resize(
                config.image_size, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )

    inp_bbox_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.image_size,
                scale=(0.9, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            # transforms.Resize(
            #     config.tgt_image_size, interpolation=InterpolationMode.BICUBIC
            # ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )

    # stable diffusion image transform
    tgt_image_transform = transforms.Compose(
        [
            transforms.Resize(
                config.tgt_image_size, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(config.tgt_image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    tgt_bbox_transform = transforms.Compose(
        [
            transforms.Resize(
                config.tgt_image_size, interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(config.tgt_image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    text_transform = BlipCaptionProcessor()

    return {
        "inp_image_transform": inp_image_transform,
        "tgt_image_transform": tgt_image_transform,
        "inp_bbox_transform": inp_bbox_transform,
        "tgt_bbox_transform": tgt_bbox_transform,
        "text_transform": text_transform,
    }


def unwrap_dist_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)

    from accelerate import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
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

    print(args)
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # if (
    #     args.train_text_encoder
    #     and args.gradient_accumulation_steps > 1
    #     and accelerator.num_processes > 1
    # ):
    #     raise ValueError(
    #         "Gradient accumulation is not supported when training the text encoder in distributed training. "
    #         "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
    #     )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # =============== model ==================
    processors = create_transforms(config)
    model = ZeroBooth(config=config.model)

    # load checkpoint
    print("loading checkpoint: ", config.checkpoint)
    model.load_checkpoint(config.checkpoint)

    # optimization
    optimizer_class = torch.optim.AdamW
    model_params = model.parameters()

    optimizer = optimizer_class(
        model_params,
        lr=float(args.learning_rate),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        weight_decay=float(args.adam_weight_decay),
        eps=float(args.adam_epsilon),
    )

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
    # subject = config.subject
    image_dir = config.image_dir
    # annotation_path = os.path.join(image_dir, "annotations.json")
    # force_init_annotations = config.force_init_annotations

    # if accelerator.is_main_process:
    #     if force_init_annotations or not os.path.exists(annotation_path):
    #         print("Generating annotations...")
    #         ImageDirDataset.generate_annotations(
    #             subject=subject,
    #             image_dir=image_dir,
    #             annotation_path=annotation_path,
    #         )
    # accelerator.wait_for_everyone()

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
        subject=config.subject,
        image_dir=config.image_dir,
    )
    print(f"Loaded {len(train_dataset)} training examples")

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=sampler,
    )

    train_dataloader = IterLoader(train_dataloader)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
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

    # num_update_steps_per_epoch = math.ceil(
    #     len(train_dataloader) / args.gradient_accumulation_steps
    # )
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers("dreambooth", config=vars(args))
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    save_to = os.path.join(args.output_dir, f"{global_step}")

    validate(
        model=unwrap_dist_model(model),
        transforms=processors,
        out_dir=os.path.join(save_to, "out_images"),
        rank=accelerator.process_index,
        args=args,
    )

    # for epoch in range(args.num_train_epochs):
    model.train()

    # for step, batch in enumerate(train_dataloader):
    while True:
        batch = next(train_dataloader)

        loss = model(batch)

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model_params, args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

        if global_step % args.logging_steps == 0:
            print(logs)

        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        # Create the pipeline using using the trained modules and save it.
        if global_step % args.save_steps == 0:
            print(f"Saving model at step {global_step}.")
            save_to = os.path.join(args.output_dir, f"{global_step}")

            if hasattr(args, "save_model") and args.save_model:
                if accelerator.is_main_process:
                    unwrap_dist_model(model).save_checkpoint(save_to, accelerator)

            validate(
                model=unwrap_dist_model(model),
                transforms=processors,
                out_dir=os.path.join(save_to, "out_images"),
                rank=accelerator.process_index,
                args=args,
            )

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


# def get_val_dataset():
#     img_paths = [
#         "/export/home/workspace/dreambooth/diffusers/data/cybertrucks/1.png",
#         "/export/home/workspace/dreambooth/diffusers/data/cybertrucks/1.png",
#         "/export/home/workspace/dreambooth/diffusers/data/cybertrucks/1.png",
#     ]

#     subj_names = [
#         "truck",
#         "truck",
#         "truck",
#         # "dog",
#     ]

#     prompts = [
#     #     # "a dog swimming in the ocean, the dog is",
#         "a truck " + ", ".join(["in the mountain"] * 20),
#     #     # "a dog at the grand canyon, photo by National Geographic, the dog is",
#         "a truck "
#         +", ".join(["at the grand canyon, photo by National Geographic"] * 20),
#         "a truck "
#         +", ".join(["in an ocean"] * 20),
#     #     # "a dog wearing a superman suit, the dog is",
#     #     # "a dog wearing sunglasses, the dog is",
#     #     # "a dog " + ", ".join(["wearing a superman suit"] * 20),
#     #     # "a dog " + ", ".join(["wearing a sunglasses"] * 20),
#     ]

#     return img_paths, subj_names, prompts

# def get_val_dataset():
#     img_paths = [
#         "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#         "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#         "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#         "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#         "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#     ]

#     subj_names = [
#         "dog",
#         "dog",
#         "dog",
#         "dog",
#         "dog",
#     ]

#     prompts = [
#         # "a dog swimming in the ocean, the dog is",
#         "a dog " + "painting by van gogh, ".join([""] * 20),
#         # "a dog at the grand canyon, photo by National Geographic, the dog is",
#         "a dog "
#         +", ".join(["at the grand canyon"] * 20),
#         # "a dog wearing a superman suit, the dog is",
#         # "a dog wearing sunglasses, the dog is",
#         "a dog " + ", ".join(["at a wood doghouse"] * 20),
#         "a dog " + ", ".join(["in a bucket"] * 20),
#         "a dog " + ", ".join(["near a river"] * 20),
#     ]

#     return img_paths, subj_names, prompts

def get_val_dataset(img_path, subj_name, prompts):
    img_paths = [img_path] * len(prompts)
    subj_names = [subj_name] * len(prompts)

    prompts_new = []

    for prompt in prompts:
        prompts_new.append("a " + subj_name + ",".join([prompt] * 20))
    
    prompts = prompts_new

    return img_paths, subj_names, prompts


# def get_val_dataset():
#     img_paths = [
#         # "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#         # "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#         # "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#         # "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#         # "/export/home/workspace/dreambooth/diffusers/data/alvan-nee/alvan-nee-padded.png",
#         "/export/home/workspace/dreambooth/diffusers/data/df-bottle/df-bottle.jpg",
#         "/export/home/workspace/dreambooth/diffusers/data/df-bottle/df-bottle.jpg",
#         "/export/home/workspace/dreambooth/diffusers/data/df-bottle/df-bottle.jpg",
#         "/export/home/workspace/dreambooth/diffusers/data/df-bottle/df-bottle.jpg",
#         "/export/home/workspace/dreambooth/diffusers/data/df-bottle/df-bottle.jpg"
#     ]

#     subj_names = [
#         "bottle",
#         "bottle",
#         "bottle",
#         "bottle",
#     ]

#     prompts = [
#         # "a dog swimming in the ocean, the dog is",
#         "a bottle " + "painting by van gogh, ".join([""] * 20),
#         # "a dog at the grand canyon, photo by National Geographic, the dog is",
#         "a bottle "
#         +", ".join(["at the grand canyon"] * 20),
#         # "a dog wearing a superman suit, the dog is",
#         # "a dog wearing sunglasses, the dog is",
#         "a bottle " + ", ".join(["at a wood table"] * 20),
#         "a bottle " + ", ".join(["on a sunny meadow"] * 20),
#         "a bottle " + ", ".join(["near a river"] * 20),
#     ]

#     return img_paths, subj_names, prompts

def generate_annotations(config):
    subject = config.subject
    image_dir = config.image_dir
    annotation_path = os.path.join(image_dir, "annotations.json")
    force_init_annotations = config.force_init_annotations

    if force_init_annotations or not os.path.exists(annotation_path):
        print("Generating annotations...")
        ImageDirDataset.generate_annotations(
            subject=subject,
            image_dir=image_dir,
            annotation_path=annotation_path,
        )

def validate(model, transforms, out_dir, rank, args):
    negative_prompt = "over-exposed, saturated, blur, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

    from PIL import Image

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    subj_image_paths, subj_names, prompts = get_val_dataset(
        img_path=args.val_image_path,
        subj_name=args.val_subject,
        prompts=args.val_prompts,
    )
    # ctx_begin_pos = [
    #     len(
    #         model.tokenizer(
    #             prompt,
    #             padding="do_not_pad",
    #         ).input_ids
    #     )
    #     - 1 for prompt in prompts
    # ]
    ctx_begin_pos = [2 for prompt in prompts]

    model.eval()

    inp_tsfm = transforms["inp_image_transform"]
    txt_tsfm = transforms["text_transform"]

    for i, (img_path, subject, prompt) in enumerate(
        zip(subj_image_paths, subj_names, prompts)
    ):
        image = Image.open(img_path).convert("RGB")

        samples = {
            "input_images": inp_tsfm(image).unsqueeze(0).to(model.device),
            "class_names": [txt_tsfm(subject)],
            "prompt": [txt_tsfm(prompt)],
            "ctx_begin_pos": [ctx_begin_pos[i]],
        }

        for gs, theta in [
            (7.5, -1),
        ]:
            output = model.generate(
                samples,
                seed=3876998111 + int(rank),
                guidance_scale=gs,
                num_inference_steps=50,
                neg_prompt=negative_prompt,
            )

            prompt = prompt.replace(" ", "_")
            out_filename = f"{i}_{prompt[:20]}_gs={gs}_theta={theta}_rank{rank}.png"
            out_filepath = os.path.join(out_dir, out_filename)

            output[0].save(out_filepath)
        
    model.train()


if __name__ == "__main__":
    input_args = parse_args()

    config = yaml.load(open(input_args.config_path), Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)

    main(config)
