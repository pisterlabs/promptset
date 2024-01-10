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
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap, get_grouped_tokens

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
import sys
sys.path.append('Tag2Text')
from Tag2Text.models import tag2text
from Tag2Text import inference_ram
import torchvision.transforms as TS

# ChatGPT or nltk is required when using tags_chineses
# import openai
# import nltk

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
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image




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
    '''
    Output:
        boxes_filt: (num_filt, 4)
        scores: (num_filt, 256), tensor.float32
        pred_phrases: list of str, label(conf), unaligned
        pred_phrases_set: list of token token map. 
                            In each token map, there is a map of pair {token_name:'score}
    '''
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
    grouped_token_ids, posmap_to_prompt_id = get_grouped_tokens(tokenized['input_ids'], tokenlizer)
    assert len(grouped_token_ids)+1 == len(caption.split('.'))
    
    # build pred
    bbox_matrix = []
    scores = [] # num_filt, list of max score in each bbox
    pred_phrases = []   # num_filt, list of unaligned phrases {label_name:score}
    pred_phrases_set = [] # num_filt, list of token map {label_name:score}


    for logit, box in zip(logits_filt, boxes_filt):
        token_map    = {}
        max_label = ''
        max_score = 0.0

        # get phrases in input texts
        posmap     = logit > text_threshold
        prompt_ids = posmap_to_prompt_id[posmap.nonzero(as_tuple=True)[0]]
        prompt_ids = torch.unique(prompt_ids)
        assert prompt_ids.min()>=0 and prompt_ids.max()<len(grouped_token_ids)

        for prompt_id in prompt_ids:
            prompt_posmap = grouped_token_ids[prompt_id]
            prompt_tokens = [tokenized['input_ids'][i] for i in prompt_posmap]
            pred_label = tokenlizer.decode(prompt_tokens)
            pred_score = logit[prompt_posmap].max().item()
            if logit[prompt_posmap].min().item()<text_threshold:
                print('[WARN] filter {} with scores {}'.format(pred_label, logit[prompt_posmap].tolist()))
                continue
            
            token_map[pred_label] = pred_score
            
            assert '#' not in pred_label, 'find # in pred_label'
            assert pred_label in caption, 'pred label not in caption'
            
            if pred_score>max_score:
                max_label = pred_label
                max_score = pred_score
            
        # record bbox
        scores.append(max_score)
        pred_phrases_set.append(token_map)
        # pred_phrases.append(max_label+ f"({str(max_score)[:4]})")
        bbox_matrix.append(box.unsqueeze(0))   
                 
        # continue
        # get phrases from all valid tokens
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append({pred_phrase: logit.max().item()})
        # pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        # scores.append(logit.max().item())
        # posmap = logit > text_threshold

    # Debug
    assert len(boxes_filt) == len(pred_phrases_set), 'pred_phrases_set length not match with boxes_filt'
    debug = True
    if debug:    
        msg = 'Detect objects\n'
        for bbox_labels in pred_phrases_set:
            # if len(bbox_labels)>1: msg += '[Ambiguous] '
            for label, conf in bbox_labels.items():
                msg += '{}:{:.3f} '.format(label, conf)
            msg += ';'
            # if len(bbox_labels)>1:
        print(msg)

    return boxes_filt, torch.Tensor(scores), pred_phrases, pred_phrases_set


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

def show_box_tokens(box,ax,token_pairs):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    # ax.text(x0, y0, label)
    labels = ''
    for token, score in token_pairs.items():
        labels += '{}({:.3f}) '.format(token, score)
    ax.text(x0, y0, labels)

def extract_good_prompts(tags, labels):
    good_prompts = []
    label_list = []    
    # print('tags are {}'.format(tags))
    
    # print('find labels:')
    for token_map in labels:
        for label,score in token_map.items():    
            # label_name, logit = label.split('(')
            label_list.append(label)
            # print(label_name)
    # print('!!!!detections are: {}'.format(label_list))
    
    for tag in tags.split('.'):
        tag = tag.strip()
        if tag in label_list:
            good_prompts.append(tag)
    if len(good_prompts)>0:
        print('find good prompts:{}'.format(good_prompts))
    return good_prompts

def add_prev_prompts(prev_prompts, tags, invalid_augment_opensets):
    augmented_tags = tags
    if tags =='':return augmented_tags
    
    for prompt in prev_prompts:
        if tags.find(prompt) == -1 and prompt not in invalid_augment_opensets:
            augmented_tags += '. ' + prompt
    return augmented_tags    
    

def save_mask_data(output_dir,frame_name, raw_tags, tags, mask_list, box_list, label_list, label_mapper):
    value = 0  # 0 for background

    mask_img = torch.zeros((len(mask_list),mask_list.shape[-2],mask_list.shape[-1])) # (K, H, W)
    for idx, mask in enumerate(mask_list):
        mask_img[idx,mask.cpu().numpy()[0] == True] = 1 #value + idx + 1
    np.save(os.path.join(output_dir, '{}_mask.npy'.format(frame_name)), mask_img.numpy())
    
    mask_instances_img = np.zeros((mask_list.shape[-2],mask_list.shape[-1]),np.uint8) # (H, W)
    json_data = {
        'raw_tags':raw_tags,
        'tags': tags,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for token_map, box, mask in zip(label_list,box_list,mask_list):
        value += 1
        json_data['mask'].append({
            'value': value,
            'labels': token_map,
            # 'logit': float(logit),
            'box': box.numpy().tolist(),
        })
        if value>0:
            mask_instances_img[mask.cpu().numpy()[0] == True] = value
    cv2.imwrite(os.path.join(output_dir, '{}_mask.png'.format(frame_name)), mask_instances_img)
    
    with open(os.path.join(output_dir, '{}_label.json'.format(frame_name)), 'w') as f:
        json.dump(json_data, f)

def read_tags_jsonfile(dir):
    tags = []
    with open(dir, 'r') as f:
        data = json.load(f)
        for item in data['mask']:
            if 'labels' in item:
                for name, score in item['labels'].items():
                    tags.append(name)
        return tags

def convert_tags(tags,mapper, composite_mapper):
    valid_tags = ''
    valid_tag_list = []
    tag_list = []
    for tag in tags.split('.'):
        tag = tag.strip()
        if tag not in tag_list: tag_list.append(tag)

    # select valid tags
    for tag in tag_list:        
        if tag in mapper and tag not in valid_tag_list: #valid_tags.find(mapper[tag]) == -1:
            valid_tags += tag + '. '
            valid_tag_list.append(tag)
        continue
        if tag in composite_mapper and composite_mapper[tag][0] in tag_list:
                valid_tags += composite_mapper[tag][1] + '. '
                valid_tag_list.append(composite_mapper[tag][1])
            
    return valid_tags[:-2]

def convert2_baseline_tags(tags,mapper):
    '''
    Convert the open-set tags to the pre-defined close-set tags
    '''
    # valid_tags = ''
    # valid_tag_list = []
    input_tag_list = []
    
    close_set_tags = ''
    close_set_tag_list =[]
    
    for tag in tags.split(','):
        tag = tag.strip()
        if tag not in input_tag_list: input_tag_list.append(tag)

    # select valid tags
    for tag in input_tag_list:        
        if tag in mapper and mapper[tag] not in close_set_tag_list: 
            # valid_tags += tag + '. '
            # valid_tag_list.append(tag)
            
            close_set_tags += mapper[tag] + '. '
            close_set_tag_list.append(mapper[tag])
  
    return close_set_tags[:-1]

def read_scans(dir):
    with open(dir, 'r') as f:
        scans = []
        for line in f.readlines():
            scans.append(line.strip())
        f.close()
        return scans
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--dataroot", type=str, required=True, help="path to data root")
    parser.add_argument("--split_file", type=str, help="name of the split file")
    # parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", required=True, type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    # parser.add_argument(
    #     "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    # )
    parser.add_argument("--frame_gap", type=int, default=1, help="skip frames")
    # parser.add_argument("--filter_tag",action='store_true', help="filter predefined tag")
    parser.add_argument("--tag_mode",type=str,default='proposed', help="raw, close_set, proposed")
    parser.add_argument("--reverse_prediction",action='store_true', help="Predict from the last frame")
    parser.add_argument("--augment_off",action='store_true', help="Turn off bi-directional prompts augmentation")
    parser.add_argument("--unaligned_phrases",action="store_true", help="use the original unaligned phrases")
    
    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--dataset",type=str, default='scannet', help="scannet or tum")
    
    args = parser.parse_args()
    print('tag mode: {}'.format(args.tag_mode))    

    import check_category
    category_file = './categories_new.json'
    root_category, mapper, composite_mapper = check_category.read_category(category_file)
    
    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    # input_folder = args.input_image
    dataroot = args.dataroot
    split_file = args.split_file
    openai_key = args.openai_key
    split = args.split
    openai_proxy = args.openai_proxy
    # output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    
    if args.dataset == 'scannet':
        RGB_FOLDER_NAME = 'color'
        RGB_POSFIX = '.jpg'
    else:
        raise NotImplementedError
    
    # PRESET_VALID_AUGMENTATION = ['wall','tile wall','cabinet','door','bookshelf','shelf','fridge']
    invalid_augment_opensets = ['floor','carpet','door','glass door','window','fridge','refrigerator']
    # for os_name, nyu_name in mapper.items():
    #     if nyu_name in PRESET_VALID_AUGMENTATION:
    #         valid_augment_opensets.append(os_name)
    print('invalid augment opensets: {}'.format(invalid_augment_opensets))
    # exit(0)
    
    AUGMENT_WINDOW_SIZE = 5
    print('dataset:{}, split:{}, scan file:{}'.format(args.dataset,split, split_file))
    # exit(0)
    
    import time, glob
    
    # ChatGPT or nltk is required when using tags_chineses
    # openai.api_key = openai_key
    # if openai_proxy:
        # openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    # make dir
    # os.makedirs(output_dir, exist_ok=True)

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    
    # load ram
    ram_model = tag2text.ram(pretrained=ram_checkpoint,
                                        image_size=384,
                                        vit='swin_l')
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()
    ram_model = ram_model.to(device)

    # Dataset
    if split_file==None:
        scans = os.listdir(dataroot)
        print('find {} scans at {}'.format(len(scans),dataroot))
        # print(scans[0])
    else:
        scans = read_scans(os.path.join(dataroot,'splits', split_file + '.txt'))
        print('find {} scans'.format(len(scans)))
    # exit(0)
    # scans = ['scene0329_00']
    # scans = ['scene0277_00','scene0670_01'] # ['scene0552_00','scene0334_00']
    # scans = ['255']
    
    ram_time = []
    dino_time = []
    sam_time = []
    total_time = []
    frame_numer = 0
    
    for scan in scans:
        scene_dir = os.path.join(dataroot,split, scan)
        input_folder = os.path.join(scene_dir, RGB_FOLDER_NAME)
        imglist = glob.glob(os.path.join(input_folder, '*{}'.format(RGB_POSFIX)))
        if args.reverse_prediction:
            imglist = sorted(imglist, reverse=True)
            output_dir = os.path.join(scene_dir, 'prediction_backward')
        else: 
            imglist = sorted(imglist)
            output_dir = os.path.join(scene_dir, 'prediction_vaug5')
        if args.augment_off:
            output_dir = os.path.join(scene_dir, 'prediction_no_augment')
        if args.unaligned_phrases:
            output_dir = os.path.join(scene_dir, 'prediction_unaligned_phrases')

        if os.path.exists(output_dir)==False:
            os.makedirs(output_dir)
        print(input_folder)
        print(RGB_POSFIX)
        print('---------- Run {}, {} imgs ----------'.format(scan, len(imglist)))
        prev_detected_prompts = []
        future_detected_prompts = [] # used in reverse prediction
        
        # load image
        for i, image_path in enumerate(imglist):
            img_name = image_path.split('/')[-1][:-4]
            if args.dataset =='scannet':
                frame_idx = int(img_name.split('-')[-1])
            elif args.dataset =='fusionportable':
                frame_idx = int(img_name.split('.')[0])
            elif args.dataset == 'tum': # tum
                frame_idx = i
                # frame_idx = float(img_name) #int(img_name.split('.')[0])
            elif args.dataset == 'scenenn':
                frame_idx = int(img_name[-5:])
            elif args.dataset =='realsense':
                frame_idx = i
            else:
                raise NotImplementedError
            
            # if frame_idx >2420:break
            if args.augment_off and frame_idx % args.frame_gap !=0:
                continue
            if os.path.exists(os.path.join(output_dir, '{}_label.json'.format(img_name))):
                continue
            # if i>10: break
            # if frame_idx != 975: continue

            print('--- processing ---', img_name)
            print(frame_idx)
            
            image_pil, image = load_image(image_path)

            # run ram
            raw_image = image_pil.resize(
                            (384, 384))
            raw_image  = transform(raw_image).unsqueeze(0).to(device)
            
            t_start = time.time()
            res = inference_ram.inference(raw_image , ram_model) #[tags, tags_chinese]
            t_ram = time.time() - t_start
            
            # Process tags
            ram_tags=res[0].replace(' |', '.')
            if args.tag_mode=='proposed':
                valid_tags = convert_tags(ram_tags, mapper, composite_mapper)
            elif args.tag_mode=='raw':
                valid_tags = ram_tags
            # elif args.tag_mode =='close_set':
            #     valid_tags = convert2_baseline_tags(ram_tags, mapper)
            else:
                raise NotImplementedError
                
            print("valid Tags: ", valid_tags)
            if args.augment_off==False:
                for prev_tag in reversed(prev_detected_prompts):
                    if abs(frame_idx - prev_tag['frame'])>AUGMENT_WINDOW_SIZE: break
                    valid_tags = add_prev_prompts(prev_tag['good_prompts'],valid_tags, invalid_augment_opensets)

                if args.reverse_prediction:
                    debug_tmp = []
                    for future_tag in reversed(future_detected_prompts):
                        if abs(frame_idx - future_tag['frame'])>AUGMENT_WINDOW_SIZE: break
                        valid_tags = add_prev_prompts(future_tag['good_prompts'],valid_tags, invalid_augment_opensets)    
                        debug_tmp.append(future_tag['frame'])
                    print('add future tags: {}'.format(debug_tmp))
                print('aug   Tags:{}'.format(valid_tags))
            else:
                print('propmts augmentation is off!')

            # run grounding dino model
            if len(valid_tags.split('.'))<1 or len(valid_tags)<2: 
                print('skip {} due to empty tags'.format(img_name))
                continue
            
            boxes_filt, scores, pred_phrases_unaligned, pred_phrases_set = get_grounding_output(
                model, image, valid_tags, box_threshold, text_threshold, device=device
            )

            t_grounding = time.time() - t_start - t_ram
            if boxes_filt.size(0) == 0:
                continue
                
            # continue
            # run SAM
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2 # topleft
                boxes_filt[i][2:] += boxes_filt[i][:2] # bottomright

            # Check and merge overlapped boxes
            # merge_pairs = {}
            # for i in range(boxes_filt.size(0)-1):
            #     for j in range(i+1,boxes_filt.size(0)):
            #         iou = torchvision.ops.box_iou(boxes_filt[i].unsqueeze(0), boxes_filt[j].unsqueeze(0))
            #         # print('iou:{}'.format(iou))
            #         if iou>0.98:
            #             if i not in merge_pairs:
            #                 merge_pairs[j] = i
            #             else: merge_pairs[j] = merge_pairs[i]
            # for child_id, parent_id in merge_pairs.items():
            #     for oslabel, score in pred_phrases_set[child_id].items():
            #         if oslabel not in pred_phrases_set[parent_id]:
            #             pred_phrases_set[parent_id][oslabel] = score
            
            # boxes_filt = boxes_filt[[i for i in range(boxes_filt.size(0)) if i not in merge_pairs]]
            # pred_phrases_set = [pred_phrases_set[i] for i in range(len(pred_phrases_set)) if i not in merge_pairs]
            # scores = scores[[i for i in range(scores.size(0)) if i not in merge_pairs]]
            # if len(merge_pairs)>0:
            #     print('[TOCHECK!] {}/{} child boxes are merged'.format(len(merge_pairs), boxes_filt.size(0)+len(merge_pairs)))
        
            # Reduce the score of door be covered by [cabinet, closet, fridge]
            for i, door_token_map in enumerate(pred_phrases_set):
                if 'door' in door_token_map:
                    door_box = boxes_filt[i]
                    
                    for j in range(boxes_filt.size(0)):
                        if i==j: continue
                        if 'cabinet' in pred_phrases_set[j] or 'closet' in pred_phrases_set[j] or 'fridge' in pred_phrases_set[j]:
                            cabinet_box = boxes_filt[j]
                            iou = torchvision.ops.box_iou(door_box.unsqueeze(0), cabinet_box.unsqueeze(0))
                            if iou>0.8:
                                scores[i] = scores[j]-0.1
                                break
        
            # Filter overlapped bbox using NMS
            boxes_filt = boxes_filt.cpu()
            tmp_count_bbox = boxes_filt.shape[0]
            nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases_unaligned = [pred_phrases_unaligned[idx] for idx in nms_idx]
            pred_phrases_set = [pred_phrases_set[idx] for idx in nms_idx]
            print('After NMS, {}/{} bbox are valid'.format(len(nms_idx), tmp_count_bbox))
            # print(f"After NMS: {boxes_filt.shape[0]} boxes")
            # tags_chinese = check_tags_chinese(tags_chinese, pred_phrases)
            # print(f"Revise tags_chinese with number: {tags_chinese}")

            # Filter out boxes too large
            size_idx = []
            for i in range(boxes_filt.size(0)):
                box_width = (boxes_filt[i][2] - boxes_filt[i][0])/W
                box_height = (boxes_filt[i][3] - boxes_filt[i][1])/H
                    
                if box_width>0.95 and box_height>0.95:
                    # print('filter too large bbox {}'.format(list(pred_phrases_set[i].keys())[0]))
                    continue
                else: size_idx.append(i)
            print('{}/{} bboxes are valid after size filtering'.format(len(size_idx), boxes_filt.size(0)))
            boxes_filt = boxes_filt[size_idx]
            pred_phrases_unaligned = [pred_phrases_unaligned[idx] for idx in size_idx]
            pred_phrases_set = [pred_phrases_set[idx] for idx in size_idx]

            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
            if transformed_boxes.size(0) == 0:
                continue
            
            # Update good prompts in adjacent frames
            good_prompts = [prompt for prompt in extract_good_prompts(valid_tags, pred_phrases_set) if prompt in ram_tags]
            prev_detected_prompts.append(
                {'frame':frame_idx,"good_prompts":good_prompts}
            )
            
            if args.reverse_prediction: # load from forward result
                forward_res_dir = os.path.join(scene_dir, 'prediction_forward', '{}_label.json'.format(img_name))
                if os.path.exists(forward_res_dir):
                    good_prompts = [prompt for prompt in read_tags_jsonfile(forward_res_dir) if prompt in ram_tags]
                    future_detected_prompts.append(
                        {'frame':frame_idx,"good_prompts": good_prompts}
                    )
                # print('save {} forward result:{}'.format(forward_res_dir,read_tags_jsonfile(forward_res_dir)))
            
            # print('save prompts:{}'.format(prev_detected_prompts))

            if frame_idx % args.frame_gap != 0:
                continue
            
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
            )
            t_sam = time.time() - t_start - t_ram - t_grounding
            t_total = time.time() - t_start
            print('RAM:{:.2f}s, Grounding:{:.2f}s, SAM:{:.2f}s, total:{:.2f}s'.format(
                t_ram, t_grounding, t_sam, t_total))
            ram_time.append(t_ram)
            dino_time.append(t_grounding)
            sam_time.append(t_sam)
            total_time.append(t_total)
            frame_numer +=1

            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                
            if args.unaligned_phrases:
                phrases_to_save = pred_phrases_unaligned
                # for box, label in zip(boxes_filt, pred_phrases_unaligned):
                #     show_box(box.numpy(), plt.gca(), label)
            else: # aligned prompts
                phrases_to_save = pred_phrases_set
                
            for box, token_map in zip(boxes_filt, phrases_to_save):
                show_box_tokens(box.numpy(), plt.gca(), token_map)
                    
            plt.title('RAM-tags:' + ram_tags + '\n' + 'filter tags:'+valid_tags+'\n') # + 'RAM-tags_chineseing: ' + tags_chinese + '\n')
            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, "{}_det.jpg".format(img_name)), 
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )

            save_mask_data(output_dir, img_name, ram_tags, valid_tags, masks, boxes_filt, phrases_to_save, mapper)
            
            # break

        print('--finished {}'.format(scan))
        # break
        
    # Summarize time
    ram_time = np.array(ram_time)
    dino_time = np.array(dino_time)
    sam_time = np.array(sam_time)
    total_time = np.array(total_time)
    out_time_file = os.path.join(dataroot, 'output','autolabel_{}.log'.format(args.split_file))
    with open(out_time_file,'w') as f:
        f.write('{} scans, frame gap {}, {} frames \n'.format(len(scans),args.frame_gap, frame_numer))
        f.write('enable multiple token for each detection. identity boxes are merged. \n')
        f.write('filter out boxes that are too large \n')
        f.write('filter tag: {} \n'.format(args.tag_mode))
        f.write('average ram time:{:.3f} ms, dinotime:{:.3f} ms, samtime:{:.3f} ms, totaltime:{:.3f} ms\n'.format(
            1000*np.mean(ram_time), 1000*np.mean(dino_time), 1000*np.mean(sam_time), 1000*np.mean(total_time)))
        f.write('total time:{:.2f} s'.format(np.sum(total_time)))
        f.close()
