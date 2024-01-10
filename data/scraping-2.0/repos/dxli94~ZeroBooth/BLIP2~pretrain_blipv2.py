'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from constant import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from models.blipv2_bert import blip_pretrain
import utils
from utils import warmup_lr_schedule, cosine_lr_schedule
from dataset import pretrain_dataset, WDSDataset

from retrieval_evaluator import retrieval_evaluator
from caption_evaluator import caption_evaluator

def train(model, data_loader, wds_data_loader, optimizer, scaler, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   

    wds_iter = iter(wds_data_loader)
    
    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
            
        optimizer.zero_grad()
        
        if config['use_laion']:
            image_wds, text_wds = next(wds_iter)
            image = torch.cat([image,image_wds],dim=0)
            text = list(text) + list(text_wds)        
        
        image = image.to(device,non_blocking=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):             
            loss_itc, loss_itm, loss_lm = model(image=image, text=text)  
            loss = loss_itc + loss_itm + loss_lm
            
        if scaler is not None:
            scaler.scale(loss).backward()        
            scaler.step(optimizer)
            scaler.update() 
        else:
            loss.backward()        
            optimizer.step()   

        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def preprocess(sample):
    image, caption = sample
    transform_train = transforms.Compose([              
            transforms.RandomResizedCrop(config['image_size'],scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),   
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ])        
    image = transform_train(image)
    caption = random.choice(caption)    
    return image, caption


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    transform_train = transforms.Compose([              
            transforms.RandomResizedCrop(config['image_size'],scale=(0.2, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),   
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ])        
    dataset = pretrain_dataset(config['train_ann'], transform_train) 
    
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True)

    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=5,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
    )  
    
    print("Creating web dataset")
    wds_dataset = WDSDataset(config['train_web'], preprocess)
    wds_data_loader = DataLoader(
        wds_dataset,
        batch_size=config['batch_size'],
        num_workers=5,
        persistent_workers=True
    )    
    
    
    #### Zero-shot retrieval evaluator ####
    print("Create zero-shot evaluator")
    transform_test = transforms.Compose([
        transforms.Resize(config['image_size'],interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ])       
    if utils.is_main_process():                         
        c_eval = caption_evaluator(transform_test, config['coco_path'], config['coco_gt_root'], args.result_dir)
    r_eval = retrieval_evaluator(transform_test, config['retrieval_image_path'], dataset=config['retrieval_dataset'])
    
    #### Model #### 
    print("Creating model")
    model = blip_pretrain(config=config)

    model = model.to(device)   

    #### Optimizer #### 
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        num_parameters += p.data.nelement()     

    optim_params = [{"params": p_wd, "weight_decay": config['weight_decay']},
                    {"params": p_non_wd, "weight_decay": 0}]    
    optimizer = torch.optim.AdamW(optim_params, lr=config['init_lr'], weight_decay=config['weight_decay'], betas=(0.9,0.98)) 
    print("number of trainable parameters: %d"%num_parameters)
    
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
        
    start_epoch = 0
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                        
        msg = model.load_state_dict(state_dict)  
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.amp:
            scaler.load_state_dict(checkpoint['scaler'])        
        start_epoch = checkpoint['epoch']+1                
        print('resume checkpoint from %s'%args.checkpoint)    
                
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module 
           
    if args.evaluate:
        retrieval_result = r_eval.evaluate_itm(model_without_ddp, device)
        print(retrieval_result)        
        if utils.is_main_process():              
            caption_result = c_eval.evaluate(model_without_ddp, device)
            print(caption_result)           
        exit()
        
    print("Start training")
    start_time = time.time()    
    for epoch in range(start_epoch, config['max_epoch']):
        data_loader.sampler.set_epoch(epoch)
        
        cosine_lr_schedule(optimizer, epoch, config['max_epoch']-1, config['init_lr'], config['min_lr'])
                
        train_stats = train(model, data_loader, wds_data_loader, optimizer, scaler, epoch, device, config) 
        
        if utils.is_main_process():    
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch)) 
            
        retrieval_result = r_eval.evaluate_itm(model_without_ddp, device)
        print(retrieval_result)        
        if utils.is_main_process():              
            caption_result = c_eval.evaluate(model_without_ddp, device)
            print(caption_result)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'{k}': v for k, v in retrieval_result.items()},      
                         **{f'{k}': v for k, v in caption_result.eval.items()},     
                         'epoch': epoch,
                        }                       
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()        
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain_blip.yaml')
    parser.add_argument('--output_dir', default='pretrain_blipv2/laion_14m_cocovg/openclip_query32_dec_e15') 
    parser.add_argument('--checkpoint', default='pretrain_blipv2/laion_14m_cocovg/openclip_query32_dec_e15/checkpoint_12.pth')    
    parser.add_argument('--evaluate', action='store_true')  
    parser.add_argument('--amp', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)