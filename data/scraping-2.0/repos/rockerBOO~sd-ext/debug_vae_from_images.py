# Original from https://gist.github.com/Poiuytrezay1/db6b98672675456bed39d45077d44179
# Credit to Poiuytrezay1

import argparse
import os
from collections import defaultdict
from pathlib import Path

import cv2 as cv

try:
    import library.model_util as model_util
    import library.train_util as train_util
    import library.sdxl_train_util as sdxl_train_util
except ModuleNotFoundError:
    print(
        "Requires to be with the Kohya-ss sd-scripts"
        + " https://github.com/kohya-ss/sd-scripts"
    )
    print("Copy this script into your Kohya-ss sd-scripts directory")
    import sys

    sys.exit(2)

import numpy as np
import torch
import tqdm
from PIL import Image, features
from torchvision import transforms

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def load_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    img = np.array(image, np.uint8)
    return img, image.info


def process_images_group(vae, images_group):
    with torch.no_grad():
        # Stack the tensors from the same size group
        img_tensors = torch.stack(images_group, dim=0).to(vae.device)

        # Encode and decode the images
        latents = vae.encode(img_tensors).latent_dist.sample()

    return latents


def process_latents_from_images(vae, input_file_or_dir, output_dir, args):
    if args.consistency_decoder:
        from consistencydecoder import ConsistencyDecoder

        decoder_consistency = ConsistencyDecoder(device=vae.device)

    input = Path(input_file_or_dir)
    output = Path(output_dir)

    os.makedirs(str(output.absolute()), exist_ok=True)

    if input.is_dir():
        image_files = [
            file
            for file in input.iterdir()
            if file.suffix
            in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"]
        ]
    else:
        image_files = [input]

    size_to_images = defaultdict(list)
    file_names = []  # List to keep track of file names

    for image_file in image_files:
        image, _ = load_image(image_file)
        transformed_image = IMAGE_TRANSFORMS(image)
        size_to_images[transformed_image.shape[1:]].append(transformed_image)
        file_names.append(image_file)  # Save the file name

    total_images = len(file_names)
    # batch_size = args.batch_size
    batch_size = 1
    print("Temporarily limiting batch size to 1")

    vae_name = Path(args.vae).stem if args.vae is not None else None

    with tqdm.tqdm(total=total_images) as progress_bar:
        for i, (size, images_group) in enumerate(size_to_images.items()):
            batch_file_names = file_names[i : i + batch_size]
            print(batch_file_names)

            # Get the batch file names
            latents = process_images_group(vae, images_group)

            if args.consistency_decoder:
                consistencydecoder_and_save(
                    decoder_consistency,
                    latents,
                    batch_file_names,
                    output,
                    device=vae.device,
                )
            else:
                decode_vae_and_save(
                    vae,
                    latents,
                    batch_file_names,
                    output,
                    gif=args.gif,
                    vae_name=vae_name,
                    apng=args.apng,
                    webp=args.webp,
                    mp4=False,
                )

            progress_bar.update(1)


def decode_vae_and_save(
    vae,
    latents,
    filenames,
    output,
    gif=False,
    vae_name=None,
    apng=False,
    webp=False,
    mp4=False,
):
    with torch.no_grad():
        decoded_images = []
        for i in range(0, 1):
            decoded_images.append(
                vae.decode(
                    latents[i : i + 1] if i > 1 else latents[i].unsqueeze(0)
                ).sample
            )
        decoded_images = torch.cat(decoded_images)

    # Rescale images from [-1, 1] to [0, 255] and save
    decoded_images = (
        ((decoded_images / 2 + 0.5).clamp(0, 1) * 255)
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
        .astype("uint8")
    )

    vae_file_part = f"-{vae_name}" if vae_name is not None else ""

    for i, decoded_image in enumerate(decoded_images):
        original_file = filenames[
            i
        ]  # Get the original file name for each image
        print(original_file)

        output_file = (
            output.absolute()
            / original_file.with_name(
                f"{original_file.stem}-latents-decoded{vae_file_part}.png"
            ).name
        )

        output_image = Image.fromarray(decoded_image)
        print(f"Saving to {output_file}")
        output_image.save(output_file)

        if gif or apng or webp:
            original_image = Image.open(original_file)

        if gif:
            output_gif_file = (
                output.absolute()
                / original_file.with_name(
                    f"{original_file.stem}-latents-decoded{vae_file_part}.gif"
                ).name
            )

            print(f"Saving gif to {output_gif_file}")
            print([original_file, output_file])
            original_image.save(
                output_gif_file,
                save_all=True,
                append_images=[output_image],
                optimize=False,
                duration=500,
                loop=0,
            )

        if mp4:
            output_mp4_file = (
                output.absolute()
                / original_file.with_name(
                    f"{original_file.stem}-latents-decoded{vae_file_part}.mp4"
                ).name
            )

            print(f"Saving mp4 to {output_mp4_file}")

            width, height = original_image.size
            # fourcc = cv.VideoWriter_fourcc(*"mp4v")
            fps = 2
            video = cv.VideoWriter(
                str(output_mp4_file), -1, fps, (width, height)
            )

            open_cv_image = np.array(original_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            video.write(open_cv_image)

            open_cv_image = np.array(output_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            video.write(open_cv_image)

            cv.destroyAllWindows()
            video.release()

        if apng:
            output_apng_file = (
                output.absolute()
                / original_file.with_name(
                    f"{original_file.stem}-latents-decoded{vae_file_part}.apng"
                ).name
            )

            print(f"Saving animated png to {output_apng_file}")
            print([original_file, output_file])
            original_image.save(
                output_apng_file,
                save_all=True,
                append_images=[output_image],
                duration=500,
                loop=0,
            )

        if webp:
            if features.check("webp_anim"):
                output_webp_file = (
                    output.absolute()
                    / original_file.with_name(
                        f"{original_file.stem}-latents-decoded{vae_file_part}.webp"
                    ).name
                )

                print(f"Saving animated webp to {output_webp_file}")
                print([original_file, output_file])
                try:
                    original_image.save(
                        output_webp_file,
                        save_all=True,
                        append_images=[output_image],
                        duration=500,
                        method=4,
                        lossless=True,
                        loop=0,
                    )
                except RuntimeError as err:
                    print(f"animated webp Error: {err}")
            else:
                print("warning: animated webp images not supported")


def consistencydecoder_and_save(
    decoder_consistency, latents, filenames, output_dir, device
):
    from consistencydecoder import save_image

    with torch.no_grad():
        sample_consistences = decoder_consistency(latents)

        for i, decoded_image in enumerate(sample_consistences):
            original_file_name = filenames[i]
            # Get the original file name for each image
            original_name_without_extension = os.path.splitext(
                original_file_name
            )[0]
            save_image(
                decoded_image,
                os.path.join(
                    output_dir,
                    f"{original_name_without_extension}-latents-decoded-consistency.png",
                ),
            )


def main(args):
    device = torch.device(args.device)

    # Convert blank VAE into None for compatibility
    if args.vae == "":
        args.vae = None

    if args.vae is None:
        from accelerate import Accelerator

        accelerator = Accelerator()
        if args.sdxl:
            # putting this in here just to be able to pass the argument

            _, _, _, vae, _, _, _ = sdxl_train_util.load_target_model(
                args,
                accelerator,
                args.pretrained_model_name_or_path,
                torch.float16,
            )
        else:
            # Load model's VAE
            _, vae, _, _ = train_util.load_target_model(
                args, torch.float16, accelerator
            )
            vae.to(device, dtype=torch.float32)
    else:
        vae = model_util.load_vae(args.vae, torch.float32).to(device)

    # Save image decoded latents
    process_latents_from_images(
        vae, args.input_file_or_dir, args.output_dir, args
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--device", default="cpu")
    argparser.add_argument(
        "--input_file_or_dir",
        help="Input file or directory to load the images from",
    )
    argparser.add_argument(
        "--output_dir", help="Output directory to put the VAE decoded images"
    )
    argparser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ",
    )
    argparser.add_argument(
        "--pretrained_model_name_or_path",
        default="",
        help="Stable diffusion model name or path to load the VAE from.",
    )
    argparser.add_argument(
        "--gif",
        action="store_true",
        help="Make a gif of the decoded image with the original",
    )

    argparser.add_argument(
        "--apng",
        action="store_true",
        help="Make an animated png of the decoded image with the original",
    )

    argparser.add_argument(
        "--webp",
        action="store_true",
        help="Make an animated webp of the decoded image with the original",
    )

    argparser.add_argument(
        "--v2", action="store_true", help="Is a Stable Diffusion v2 model."
    )

    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to process the images.",
    )

    argparser.add_argument(
        "--sdxl", action="store_true", help="(NOTWORKING) SDXL model"
    )
    argparser.add_argument(
        "--lowram", type=int, default=1, help="SDXL low ram option"
    )

    argparser.add_argument(
        "--full_fp16", type=int, default=1, help="SDXL use full fp16"
    )
    argparser.add_argument(
        "--full_bf16", type=int, default=1, help="SDXL use full bf16"
    )

    argparser.add_argument(
        "--consistency_decoder",
        action="store_true",
        help="Use Consistency Decoder from OpenAI https://github.com/openai/consistencydecoder",
    )

    args = argparser.parse_args()
    main(args)
