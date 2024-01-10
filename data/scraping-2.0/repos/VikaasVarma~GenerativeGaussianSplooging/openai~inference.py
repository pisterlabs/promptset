"""
Evaluation script adapted from openai/guided-diffusion to condition on images.
Applies the image-conditional model to all images in a directory, writing the results
to an output directory.
"""

import argparse
import os

from torchvision.transforms.v2.functional import to_pil_image, to_tensor
from PIL import Image
import torch as th
import util

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    device = dist_util.dev()
    print("Loading model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        in_channels=6
    )
    print("Moving to cuda...")
    model.to(device)
    print("Loading params...")
    model.load_state_dict(
        th.load(args.model_path, map_location=lambda storage, loc: storage)
    )
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    files = os.listdir(args.data_dir)
    print("Reading from", args.data_dir)
    out_dir = os.path.join(args.out_dir, f"openai_gs={args.gs}")
    print("Writing to", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    transform = util.eval_transform(size=args.image_size)

    for start in range(0, len(files), args.batch_size):
        print(start,"/",len(files))
        end = min(start + args.batch_size, len(files))
        tensors = []
        output_paths = []
        for i in range(start, end):
            with Image.open(os.path.join(args.data_dir, files[i])) as im:
                tensor = transform(im)
                tensor = tensor.to(device) * 2 - 1
                tensors.append(tensor.unsqueeze(0))

            output_paths.append(os.path.join(out_dir, files[i]))

        cond = th.cat(tensors, dim=0)
        model_kwargs = {"concat_cond": cond}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            gs=args.gs
        )
        sample = ((sample + 1) / 2).clamp(0, 1)
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        # sample = sample.contiguous()

        n = sample.shape[0]
        for i in range(n):
            im = to_pil_image(sample[i])
            im.save(output_paths[i])


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--gs", type=float, default=2)
    return parser


if __name__ == "__main__":
    main()
