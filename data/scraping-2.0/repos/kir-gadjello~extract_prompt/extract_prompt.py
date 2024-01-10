#!/usr/bin/env python3
# Prompt Extractor v0.4
# Copyright (c) 2022 kir-gadjello, WonkyGrub, pharmapsychotic

import os
import gc
import pathlib
import time
import argparse

from PIL import Image
from tabulate import tabulate
import requests

import torch
import open_clip as clip

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from models.blip import blip_decoder

from utilities import (
    touch_model,
    ensure_file,
    chunk,
    make_weightlist_parser,
    hash_arr,
    print_once,
    load_list,
)

CACHE_DIR = "./cache"
EMBEDDING_CACHE_DIR = f"{CACHE_DIR}/embs"
MODEL_CACHE_DIR = f"{CACHE_DIR}/models"
IMG_CACHE_DIR = f"{CACHE_DIR}/imgs"

pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(EMBEDDING_CACHE_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(IMG_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def try_cache(
    key_array,
    recompute=None,
    split_chunks=None,
    combine=lambda loaded: torch.vstack(loaded),
    label="",
    cache_dir=EMBEDDING_CACHE_DIR,
    ext=".pth",
    silent=False,
    hitsize=lambda _: 1,
):
    assert recompute is not None

    def getkey(key_array):
        key = hash_arr(key_array)
        path = os.path.join(cache_dir, f"{key}{ext}")
        return path

    if split_chunks is None:
        chunks = [key_array]
    else:
        chunks = split_chunks(key_array)

    loaded = []
    N = len(chunks)
    pbar = None
    hit = 0
    i_done = 0

    for i, ch in enumerate(chunks):
        path = getkey(ch)
        if not os.path.isfile(path):
            print_once(
                "Populating cache, this can take a few minutes...",
                "pop_cache",
                printer=tqdm.write,
            )
            if pbar is None:
                pbar = tqdm(total=sum(list(map(len, chunks))) - i_done)

            t0 = time.time()
            value = recompute(ch)
            torch.save(value, path)
            t1 = time.time()

            tqdm.write(f"CACHE:{label} recomputing #{i}/{N} took {round(t1-t0, 2)}s")
            loaded.append(value)
        else:
            ret = torch.load(path)
            loaded.append(ret)
            hit += hitsize(ret)

        if pbar is not None:
            pbar.update(len(ch))

        i_done += len(ch)

    if pbar is not None:
        pbar.close()

    if not silent:
        print(f"CACHE:{label} restored {hit} items")

    return combine(loaded)


avail_models = dict(
    ViTB32=["ViT-B/32", "ViT-B-32-quickgelu", "openai"],
    ViTB32_LAION2B=[None, "ViT-B-32", "laion2b_e16"],
    ViTB16=["ViT-B/16", "ViT-B-16", "openai"],
    ViTL14=["ViT-L/14", "ViT-L-14", "openai"],
    ViTL14_LAION400M=[None, "ViT-L-14", "laion400m_e32"],
    ViTL14_336px=["ViT-L/14@336px", "ViT-L-14-336", "openai"],
    RN101=["RN101", "RN101-quickgelu", "openai"],
    RN50x4=["RN50x4", "RN50x4", "openai"],
    RN50x16=["RN50x16", "RN50x16", "openai"],
    RN50x64=["RN50x64", "RN50x64", "openai"],
)


def create_clip(mname, half=False, device=None):
    assert mname in avail_models
    _, laion_name, source = avail_models[mname]
    model, _, transform = clip.create_model_and_transforms(
        laion_name,
        pretrained=source,
        precision="fp16" if half else "fp32",
        device=device,
    )
    model.eval()
    setattr(model, "mname", mname)
    return model, transform


default_aspects = dict(mediums=1, artists=1, trending=1, movements=1, flavors=3)


parser = argparse.ArgumentParser()
parser.add_argument(
    "images", type=str, nargs="*", default=[], help="image to reverse-prompt-engineer"
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="device [cpu, cuda, mps]",
)
parser.add_argument(
    "--clip_models",
    type=str,
    default="ViTL14",
    help=f"CLIP models to use, default is ViTL14 from OpenAI. Select from [{','.join(avail_models.keys())}]",
)
parser.add_argument(
    "--caption_model",
    type=str,
    default="base",
    help="BLIP model to use, [base, large]",
)
parser.add_argument(
    "--top",
    type=make_weightlist_parser(default_aspects),
    default="",
    help="top ranks to output across aspects, 0 removes aspect",
)
parser.add_argument(
    "--half", action="store_true", help="evaluate at half precision", default=False
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="verbose logging", default=False
)
args = parser.parse_args()

print("==<{ Prompt Extractor v0.4 }>==")

used_top = {}
for k, v in args.top.items():
    used_top[k] = f"top-{v}"
asp = ",".join(list(map(lambda ab: f"{ab[0]}:{ab[1]}", list(used_top.items()))))
print(f"Using aspects: [{asp}]")

device = torch.device(args.device)

blip_models = dict(
    base=(
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth",
        896081425,
    ),
    large=(
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth",
        1785411505,
    ),
)
blip_image_eval_size = 384


def load_blip_model(args):
    name = args.caption_model
    assert name in blip_models
    url, expected_size = blip_models[name]
    fname = f"blip_model_{name}_caption.pth"

    ensure_file(url, fname, dstdir=MODEL_CACHE_DIR, size=expected_size)

    return touch_model(
        blip_decoder(
            pretrained=os.path.join(MODEL_CACHE_DIR, fname),
            image_size=blip_image_eval_size,
            vit=name,
        ),
        args,
    )


blip_model = load_blip_model(args)


def generate_caption(pil_image):
    gpu_image = (
        transforms.Compose(
            [
                transforms.Resize(
                    (blip_image_eval_size, blip_image_eval_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )(pil_image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        caption = blip_model.generate(
            gpu_image, sample=False, num_beams=3, max_length=20, min_length=5
        )
    return caption[0]


def rank(model, image_features, text_array, top_count=1, label=""):
    top_count = min(top_count, len(text_array))
    mname = model.mname

    def recompute(key_array):
        t_array = key_array[1:]
        text_tokens = clip.tokenize([text for text in t_array])
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def split_chunks(key_array):
        mname = key_array[0]
        text_items = key_array[1:]
        chunks = list(map(lambda x: [mname] + x, chunk(text_items, 16)))
        return chunks

    text_features = try_cache(
        [mname] + text_array,
        recompute=recompute,
        split_chunks=split_chunks,
        label=label,
        silent=not args.verbose,
        hitsize=lambda x: x.shape[0],
    )

    similarity = torch.zeros((1, len(text_array))).to(device)
    for i in range(image_features.shape[0]):
        similarity += (
            100.0 * image_features[i].unsqueeze(0) @ text_features.T
        ).softmax(dim=-1)
    similarity /= image_features.shape[0]

    top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
    return [
        (text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy() * 100))
        for i in range(top_count)
    ]


data_path = "./templates/"

artists = list(
    map(
        lambda artist: f"by {artist}", load_list(os.path.join(data_path, "artists.txt"))
    )
)
flavors = load_list(os.path.join(data_path, "flavors.txt"))
mediums = load_list(os.path.join(data_path, "mediums.txt"))
movements = load_list(os.path.join(data_path, "movements.txt"))
sites = load_list(os.path.join(data_path, "sites.txt"))

trending_list = [site for site in sites]
trending_list.extend(["trending on " + site for site in sites])
trending_list.extend(["featured on " + site for site in sites])
trending_list.extend([site + " contest winner" for site in sites])


# TODO generalize this code to use different extraction strategies
def extract_prompt(image, models, top_counts={}):
    caption = generate_caption(image)
    if len(models) == 0:
        print(f"\n\n{caption}")
        return

    table = []
    bests = [[("", 0)]] * 5

    for model_name in models:
        print(f"Extracting prompt with {model_name}...")
        model, preprocess = create_clip(model_name, half=args.half, device=device)

        images = preprocess(image).unsqueeze(0)
        print("Encoding image ...", end="")
        t0 = time.time()
        with torch.no_grad():
            image_features = model.encode_image(images).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        t1 = time.time()
        print(f" done in {round(t1 - t0, 2)}s")

        ranks = []

        spec = [
            ["mediums", mediums],
            ["artists", artists],
            ["trending", trending_list],
            ["movements", movements],
            ["flavors", flavors],
        ]

        print("Ranking subprompts ... ", end="")
        t0 = time.time()

        for label, wordlist in spec:
            top_count = top_counts.get(label, 0)
            if top_count > 0:
                hdr = label
                if hdr.endswith("s"):
                    hdr = hdr[:-1]
                hdr = hdr.capitalize()

                ranks.append(
                    (
                        hdr,
                        rank(
                            model,
                            image_features,
                            wordlist,
                            top_count=top_count,
                            label=label,
                        ),
                    )
                )
            else:
                print(f'\nSkipping aspect "{label}"')

        t1 = time.time()
        print(f"done in {round(t1-t0, 3)}s")

        for i in range(len(ranks)):
            confidence_sum = 0
            ranks_i = ranks[i][1]
            for ci in range(len(ranks_i)):
                confidence_sum += ranks_i[ci][1]
            if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
                bests[i] = ranks_i

        for label, r in ranks:
            row = [model_name, label]
            if len(r) > 3:
                for x in r:
                    table.append(row + [f"{x[0]} ({x[1]:0.1f}%)"])
            else:
                row.append(", ".join([f"{x[0]} ({x[1]:0.1f}%)" for x in r]))
                table.append(row)

        del model
        gc.collect()

    print(
        tabulate(
            table,
            headers=["Model", "Aspect", "Value(s)"],
            tablefmt="grid",
        )
    )

    top_flavors = ", ".join([f"{x[0]}" for x in bests[4]])
    top_medium = bests[0][0][0]

    # This prompt extraction heuristic is ad-hoc and should be rewritten
    if caption.startswith(top_medium):
        print(
            f"({model_name}) Estimated prompt:\n{caption} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {top_flavors}"
        )
    else:
        print(
            f"({model_name}) Estimated prompt:\n{caption}, {top_medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {top_flavors}"
        )


clip_models = []
for m in args.clip_models.split(","):
    assert m in avail_models
    clip_models.append(m)

print("Using CLIP models:", args.clip_models)


def load_img(image_path_or_url):
    if str(image_path_or_url).startswith("http://") or str(
        image_path_or_url
    ).startswith("https://"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert(
            "RGB"
        )
    else:
        image = Image.open(image_path_or_url).convert("RGB")

    thumb = image.copy()
    thumb.thumbnail([blip_image_eval_size, blip_image_eval_size])
    print("Loaded image:", image)

    return image


for image in args.images:
    img = load_img(image)
    extract_prompt(img, models=clip_models, top_counts=args.top)
