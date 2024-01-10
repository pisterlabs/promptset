import argparse
import os
import copy

import numpy as np
import json
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

# ChatGPT
import openai
from tqdm import tqdm
import pdb
from datasets import load_dataset, Dataset
import time

from diffusers import StableDiffusionInpaintPipeline

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # pdb.set_trace()
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def process_image(ori_image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # pdb.set_trace()
    image, _ = transform(ori_image, None)  # 3, h, w
    return ori_image, image

def generate_caption(raw_image, device):
    # unconditional image captioning
    if device == "cuda":
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    else:
        inputs = processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_tags(caption, split=',', max_tokens=100, model="gpt-3.5-turbo"):
    prompt = [
        {
            'role': 'system',
            'content': 'Extract the unique nouns in the caption. Remove all the adjectives. ' + \
                       f'List the nouns in singular form. Split them by "{split} ". ' + \
                       f'Caption: {caption}.'
        }
    ]
    response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    # sometimes return with "noun: xxx, xxx, xxx"
    tags = reply.split(':')[-1].strip()
    return tags


def check_caption(caption, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    prompt = [
        {
            'role': 'system',
            'content': 'Revise the number in the caption if it is wrong. ' + \
                       f'Caption: {caption}. ' + \
                       f'True object number: {object_num}. ' + \
                       'Only give the revised caption: '
        }
    ]
    response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    # sometimes return with "Caption: xxx, xxx, xxx"
    caption = reply.split(':')[-1].strip()
    return caption


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, caption, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'mask_{caption}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'caption': caption,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, f'label_{caption}.json'), 'w') as f:
        json.dump(json_data, f)

def image_Splicing_2(img_1, img_2, flag='x'):
    size1, size2 = img_1.size, img_2.size
    if flag == 'x':
        joint = Image.new("RGB", (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
    else:
        joint = Image.new("RGB", (size1[0], size2[1]+size1[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
    joint.paste(img_1, loc1)
    joint.paste(img_2, loc2)
    # pdb.set_trace()
    return joint

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, required=True, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    # inpaint
    parser.add_argument("--inpaint_mode", type=str, default="first", help="inpaint mode")


    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    inpaint_mode = args.inpaint_mode

    openai.api_key = openai_key
    if openai_proxy:
        openai.proxy = {"http": openai_proxy, "https": openai_proxy}
    # 0414
    openai.api_base = "https://openai.func.icu/v1"
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    # load image locally
    dataset = load_dataset("/share/sd/detect/coco")
    dataset = dataset['val'].shuffle(0).select(list(range(20)))
    # pdb.set_trace()
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # use Tag2Text can generate better captions
    # https://huggingface.co/spaces/xinyu1205/Tag2Text
    # but there are some bugs...
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    if device == "cuda":
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
    else:
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    # "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    # )
    inpaint_pth = "/share/sd/hf/ckpt/further_finetune_sd2_inpaint_canva_2w_bsz_4_2e-6/epoch_245"
    # inpaint_pth = "/share/sd/hf/ckpt/recover_zy/epoch_120"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_pth, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    # 
    info = {
        "image":[],
        "image_id":[],
        "image_prompt":[],
        "image_wo_object":[],
        "image_wo_object_prompt":[], # only support move first object
        "object_image_prompt": [],
        "object_image": [],
        "bounding_box": [],
        # "mask":[],
        "mask_image":[],
    }
    begin = time.time() 
    for d in tqdm(dataset):
        # load image
        # image_pil, image = load_image(image_path)
        img_id = d['image_id']
        info['image_id'].append(img_id)
        image_pil, image = process_image(d['image'])

        # visualize raw image
        raw_img_pth = os.path.join(output_dir, "lama", f"{d['image_id']}.png")
        image_pil.save(raw_img_pth)
        info['image'].append(raw_img_pth)

        # generate caption and tags
        caption = generate_caption(image_pil, device=device)
        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        # pdb.set_trace()
        text_prompt = generate_tags(caption, split=split)
        print(f"Caption: {caption}")
        print(f"Tags: {text_prompt}")

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )


        # image = cv2.imread(image_path)
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")
        caption = check_caption(caption, pred_phrases)
        print(f"Revise caption with number: {caption}")
        info['image_prompt'].append(caption)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        
        # draw output image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)

        # plt.title(caption)
        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(output_dir, "visual", f"auto-{caption}.jpg"), 
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )
        # only consider the first object
        save_mask_data(output_dir, caption, masks, boxes_filt, pred_phrases)
        # (crop_images, object_names) <-> (box_filt, pred_phrases)
        if not len(boxes_filt): # NULL
            continue
        crop_img = image_pil.crop(tuple(np.array(boxes_filt[0].cpu())))
        obj = pred_phrases[0].split("(")[0]
        crop_img_pth = f"/share/sd/sam/Grounded-Segment-Anything-main/outputs/objects/object-{img_id}-{obj}.png"
        mask_img_pth = f"/share/sd/sam/Grounded-Segment-Anything-main/outputs/lama/{img_id}_mask.png" # only take the first mask
        mask_img = Image.fromarray(np.array(masks[0].cpu()[0])) # only pick the first
        info['object_image'].append(crop_img_pth)
        info['object_image_prompt'].append(obj)
        info['bounding_box'].append(boxes_filt)
        # info['mask'].append(masks)
        info['mask_image'].append(mask_img_pth)
        mask_img.save(mask_img_pth)
        crop_img.save(crop_img_pth)
        # TODO: extract single object data
        # use pred_phrases as object prompts
        # time.sleep(20) # openai, 3 requests / min
        # inpainting pipeline: remove all objects
        INPAINT = True
        if INPAINT:
            # pdb.set_trace()
            if inpaint_mode == 'merge':
                masks = torch.sum(masks, dim=0).unsqueeze(0)
                masks = torch.where(masks > 0, True, False)
            mask = masks[0][0].cpu().numpy() # simply choose the first mask, which will be refine in the future release
            # boxes_filt
            img_mask = Image.new(image_pil.mode, image_pil.size, color=(0, 0, 0)) # 0-black, 255-white
            draw = ImageDraw.Draw(img_mask)
            box = np.array(boxes_filt[0]) # only use the first box
            draw.rectangle(box, fill=(255, 255, 255), width=0.1)
            # mask_pil = Image.fromarray(mask)
            mask_pil = img_mask
            
            image_pil = Image.fromarray(image)
            ori_image = image_pil
            # image_pil = image_pil.resize((512, 512))
            # mask_pil = mask_pil.resize((512, 512))
            # prompt = "A sofa, high quality, detailed"
            # inpaint_prompt = "None"
            # inpaint_prompt = "none"
            inpaint_prompt = "no text, empty text, text removal, pure background"

            image = pipe(prompt=inpaint_prompt, image=image_pil, mask_image=mask_pil).images[0]
            # image = image.resize(size)
            remove_caption = generate_caption(image, device=device)
            print(f"After remove objects, the caption is:{remove_caption}")
            inpaint_img_pth = os.path.join(output_dir, "inpaint", f"inpaint-{img_id}.jpg")
            image.save(inpaint_img_pth)
            info['image_wo_object'].append(inpaint_img_pth)
            info['image_wo_object_prompt'].append(remove_caption)
            # whole_image = image_Splicing_2(ori_image,image)
            # whole_image.save(f"/share/sd/sam/Grounded-Segment-Anything-main/outputs/debug/debug_ori_inpaint_{img_id}.jpg")
            # w,h = image_pil.size
            # draw.text((w/2,h/2),f"{caption}",fontsize=256)
            # draw.text((w+w/2,h/2),f"{remove_caption}",fontsize=256)
            time.sleep(20)
end = time.time()
print(f"Total use time:{round(end-begin,2)}s")
info = Dataset.from_dict(info)
info.save_to_disk("/share/sd/sam/Grounded-Segment-Anything-main/outputs/results/debug_results")

# info.save_to_disk("/share/sd/sam/Grounded-Segment-Anything-main/outputs/results/debug_intermediate_results")
pdb.set_trace()