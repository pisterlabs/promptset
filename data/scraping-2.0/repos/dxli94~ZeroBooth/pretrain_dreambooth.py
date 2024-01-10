import argparse
import hashlib
import itertools
import math
import os
import yaml
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from transformers.activations import QuickGELUActivation as QuickGELU

from lavis.processors.blip_processors import BlipCaptionProcessor
from modeling_clip import CtxCLIPTextModel


logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save checkpoint every save_steps steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=200,
        help="Log every logging_steps steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def create_blipv2_model_and_preprocess(
    is_train=True,
    config_path="BLIP2/configs/finetune.yaml",
    checkpoint_path="BLIP2/pretrain_blipv2/checkpoint_best.pth",
):
    from BLIP2.models.blipv2_feature_extractor import blip
    from BLIP2.constant import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    # model
    config = yaml.load(open(config_path), Loader=yaml.Loader)

    print("Creating model")
    model = blip(config=config)

    print("Loading weights")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    if not is_train:
        model.eval()

    # preprocess
    # blip image transform
    inp_image_transform = transforms.Compose(
        [
            transforms.Resize(
                config["image_size"], interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )

    # stable diffusion image transform
    tgt_image_transform = transforms.Compose(
        [
            transforms.Resize(
                config["tgt_image_size"], interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(config["tgt_image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    text_transform = BlipCaptionProcessor()

    return model, {
        "inp_image_transform": inp_image_transform,
        "tgt_image_transform": tgt_image_transform,
        "text_transform": text_transform,
    }


class ImageNetDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(
        self,
        inp_image_transform,
        tgt_image_transform,
        text_transform,
        clip_tokenizer,
        superclass_filename="data/imagenet_superclasses.txt",
        **kwargs,
    ):
        self.inner_dataset, self.classnames = self.load_imagenet_val()

        self.inp_image_transform = inp_image_transform
        self.tgt_image_transform = tgt_image_transform

        self.text_transform = text_transform

        self.clip_tokenizer = clip_tokenizer

        self.prompt = "A {}."

        self.superclass_filename = superclass_filename
        self.label2sclassnames = self.load_superclass_names()

    def __len__(self):
        return len(self.inner_dataset)

    def get_classname(self, label_id):
        return self.classnames[label_id]

    def get_superclassname(self, label_id):
        return self.label2sclassnames[label_id]

    def __getitem__(self, index):
        example = self.inner_dataset[index]

        example["input_image"] = self.inp_image_transform(example["image"])
        example["target_image"] = self.tgt_image_transform(example["image"])

        class_name = self.text_transform(self.get_classname(example["label"]))
        superclass_name = self.text_transform(self.get_superclassname(example["label"]))

        example["class_name"] = ", ".join([class_name, superclass_name])

        prompt = self.text_transform(self.prompt_from_label(example["label"]))
        example["instance_prompt_ids"] = self.clip_tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            # max_length=self.clip_tokenizer.model_max_length,
            max_length=25,
        ).input_ids

        return example

    def prompt_from_label(self, label):
        classname = self.get_superclassname(label)
        prompt = self.prompt.format(classname)

        return prompt

    def load_imagenet(self):
        from lavis.datasets.builders import load_dataset

        dataset = load_dataset("imagenet")

        return dataset

    def load_imagenet_val(self):
        # dataset = self.load_imagenet()["val"]
        dataset = self.load_imagenet()["train"]
        classnames = dataset.classnames

        return dataset, classnames

    def load_superclass_names(self):
        with open(self.superclass_filename, "r") as f:
            classnames = f.readlines()

        classnames = [c.strip() for c in classnames]

        label2sclassnames = {}
        for i, classname in enumerate(classnames):
            label2sclassnames[i] = classname

        return label2sclassnames


class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x):
        x_in = x

        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in
        x = self.LayerNorm(x)

        return x


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

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # =============== model ==================
    # ==== BLIP ====
    blip_model, processors = create_blipv2_model_and_preprocess(is_train=True)

    # ==== projection ====
    # TODO in_dim=blip_vision_width, out_dim=clip_text_width
    proj_layer = ProjLayer(
        in_dim=768, out_dim=768, hidden_dim=3072, drop_p=0.1, eps=1e-12
    )

    # ==== stable diffusion ====
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # Load models and create wrapper for stable diffusion
    text_encoder = CtxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    # print("Adding query tokens to SD CLIP tokenizer")
    # tokenizer.add_special_tokens(
    #     {"additional_special_tokens": blip_model.query_symbols}
    # )
    # text_encoder.resize_token_embeddings(len(tokenizer))

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    vae.requires_grad_(False)
    # unet, text_encoder are not tuned, but we need gradients
    # if not args.train_text_encoder:
    #     text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # setup parameters for optimizer
    params_to_optimize = [blip_model.parameters(), proj_layer.parameters()]
    if args.train_text_encoder:
        params_to_optimize.append(text_encoder.parameters())

    if args.train_unet:
        params_to_optimize.append(unet.parameters())

    params_to_optimize = itertools.chain(*params_to_optimize)

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
    )

    # ====== Dataset ======
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["target_image"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            # max_length=tokenizer.model_max_length,
            max_length=25,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,  # used by stable diffusion
            "input_images": torch.stack(  # used by blip
                [example["input_image"] for example in examples]
            ),
            "class_names": [example["class_name"] for example in examples],
        }

        return batch

    print("Loading dataset")
    train_dataset = ImageNetDataset(**processors, clip_tokenizer=tokenizer)
    print(f"Loaded {len(train_dataset)} training examples")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # TODO
    # if args.train_text_encoder:
    #     (
    #         unet,
    #         text_encoder,
    #         blip_model,
    #         optimizer,
    #         train_dataloader,
    #         lr_scheduler,
    #     ) = accelerator.prepare(
    #         unet, text_encoder, blip_model, optimizer, train_dataloader, lr_scheduler
    #     )
    # else:
    # unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     unet, optimizer, train_dataloader, lr_scheduler
    # )
    (
        blip_model,
        proj_layer,
        unet,
        text_encoder,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        blip_model,
        proj_layer,
        unet,
        text_encoder,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    if not args.train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)

    blip_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        blip_model.train()

        # if args.train_text_encoder:
        #     text_encoder.train()

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet):
                # Compute BLIP embeddings.
                # shape: (batch_size, num_queries, embed_dim), e.g. (4, 32, 768)
                blip_embeddings = blip_model(
                    # image=batch["input_images"].to(dtype=weight_dtype),
                    image=batch["input_images"],
                    text=batch["class_names"],
                )

                # projected as clip text embeddings
                ctx_embeddings = proj_layer(blip_embeddings)

                # TODO update CLIP embedding layer with projected blip embeddings
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # TODO make it configurable rather than hardcoding 2 (2 = len(["[pad]", "a"])
                encoder_hidden_states = text_encoder(
                    input_ids=batch["input_ids"],
                    ctx_embeddings=ctx_embeddings,
                    ctx_begin_pos=2,
                )[0]

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # params_to_clip = (
                    #     itertools.chain(unet.parameters(), text_encoder.parameters())
                    #     if args.train_text_encoder
                    #     else unet.parameters()
                    # )
                    params_to_clip = itertools.chain(
                        blip_model.parameters(), proj_layer.parameters()
                    )

                    total_norm = 0
                    for p in params_to_clip:
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    print("total_norm", total_norm)

                    params_to_clip = itertools.chain(
                        blip_model.parameters(), proj_layer.parameters()
                    )

                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
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
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                print(f"Saving model at step {global_step}.")
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    revision=args.revision,
                )
                pipeline.save_pretrained(args.output_dir + f"/{global_step}")
                # save blip model and proj weights
                blip_without_ddp = accelerator.unwrap_model(blip_model)
                proj_without_ddp = accelerator.unwrap_model(proj_layer)

                blip_save_to = args.output_dir + f"/{global_step}/blip_model"
                proj_save_to = args.output_dir + f"/{global_step}/proj_layer"

                if not os.path.exists(blip_save_to):
                    os.makedirs(blip_save_to)
                if not os.path.exists(proj_save_to):
                    os.makedirs(proj_save_to)

                torch.save(
                    blip_without_ddp.state_dict(), blip_save_to + "/blip_weight.pt"
                )
                torch.save(
                    proj_without_ddp.state_dict(), proj_save_to + "/proj_weight.pt"
                )

                # pipeline.save_pretrained(args.output_dir)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
