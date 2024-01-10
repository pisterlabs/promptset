import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import openai
import pandas as pd
import argparse
import random
import requests
from tqdm import tqdm
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
import concurrent.futures
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from transformers import T5Tokenizer, T5ForConditionalGeneration

from multimodal_eval_main.modeling import InstructBlipT5Model, BlipModel
from multimodal_eval_main.data_loading import MultimodalPart, MultimodalSequence
from multimodal_eval_main.modeling import FromageModel
from multimodal_eval_main.modeling import OpenFlamingoModel
import re

##Lavin
import sys
import time
from typing import Tuple
import json
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel as LaVIN_initialize_model_parallel
from multimodal_eval_main.models.LaVIN.lavin.eval_model import ModelArgs as LaVIN_ModelArgs, Transformer as LaVIN_Transformer
from multimodal_eval_main.models.LaVIN.lavin.tokenizer import Tokenizer as LaVIN_Tokenizer
from multimodal_eval_main.models.LaVIN.lavin.generator import LaVIN_Generator
from multimodal_eval_main.models.LaVIN.lavin.mm_adapter import set_MMAdapter as LaVIN_set_MMAdapter,set_Clip_Adapter as LaVIN_set_Clip_Adapter
from multimodal_eval_main.models.LaVIN.util.apply_delta import apply_model_delta_online
from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

##for lynx_llm
from multimodal_eval_main.models.lynx_llm.models.lynx import LynxBase
import ruamel.yaml as yaml
from multimodal_eval_main.models.lynx_llm.dataset import create_dataset, create_loader
import csv
import datetime

## for mmgpt
from multimodal_eval_main.models.Multimodal_GPT.app import Inferencer as mmgpt_Inferencer

## for llama
from multimodal_eval_main.models.llama_recipes.inference.model_utils import load_model as load_model_for_llama, load_peft_model as load_peft_model_for_llama
from transformers.models.llama.tokenization_llama import LlamaTokenizer


## for mplug_owl
from transformers import AutoTokenizer
from multimodal_eval_main.models.mPLUG_Owl.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from multimodal_eval_main.models.mPLUG_Owl.mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

##miniGPT4
from multimodal_eval_main.models.MiniGPT4.minigpt4.common.config import Config as minigpt4_Config
from multimodal_eval_main.models.MiniGPT4.minigpt4.common.dist_utils import get_rank
from multimodal_eval_main.models.MiniGPT4.minigpt4.common.registry import registry as miniGPT4_registry
from multimodal_eval_main.models.MiniGPT4.minigpt4.conversation.conversation import Chat as minigpt4_Chat, CONV_VISION as minigpt4_CONV_VISION

## for llama_adapter
from multimodal_eval_main.models.LLaMA_Adapter import llama as llama_adapter
import cv2

## for vpgtrans
from multimodal_eval_main.models.VPGTrans.lavis.common.config import Config as VPGTans_Config
from  multimodal_eval_main.models.VPGTrans.lavis.common.dist_utils import get_rank
from  multimodal_eval_main.models.VPGTrans.lavis.common.registry import registry as vpgtrans_registry
from multimodal_eval_main.models.VPGTrans.lavis.conversation.conversation import Chat as VPGTans_Chat, CONV_VISION as VPGTans_CONV_VISION

# imports modules for registration
from multimodal_eval_main.models.VPGTrans.lavis.datasets.builders import *
from multimodal_eval_main.models.VPGTrans.lavis.models import *
from multimodal_eval_main.models.VPGTrans.lavis.processors import *
from multimodal_eval_main.models.VPGTrans.lavis.runners import *
from multimodal_eval_main.models.VPGTrans.lavis.tasks import *

# llava
from multimodal_eval_main.models.LLaVA.llava.conversation import conv_templates, SeparatorStyle
from multimodal_eval_main.models.LLaVA.llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from multimodal_eval_main.models.LLaVA.llava.model import *
from multimodal_eval_main.models.LLaVA.llava.model.utils import KeywordsStoppingCriteria


## chatgpt
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"




os.environ['TOKENIZERS_PARALLELISM'] = 'false'
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="Number of samples to use, better under 3")
    parser.add_argument("--setting", type=str, default="zero-shot", help="[zero-shot, few-shot, majority, random]")
    parser.add_argument("--seed", type=int, default=42, help="[0, 1, 42]")
    parser.add_argument("--shots", type=int, default=-1, help="[1, 5, 10]")
    parser.add_argument('--use_api', action='store_true', help='use api or not')
    parser.add_argument("--api", type=str, default=None, help="api key")
    parser.add_argument("--selected_tasks", type=str, default=None, help="list of string of tasks, e.g '[\"sc\"]'")
    parser.add_argument("--selected_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--ignored_datasets", type=str, default=None, help="list of string of datasets")
    parser.add_argument("--model_name", type=str, default="blip2_t5", help="[blip2_t5, blip2_vicuna_instruct, instructblip]")
    parser.add_argument("--model_type", type=str, default=None, help="[pretrain_flant5xxl, vicuna7b, flant5xxl]")
    parser.add_argument("--skip_runned", action="store_true", help="skip runned dataset")
    parser.add_argument("--root_path", type=str, default="multimodal_data", help="the path of multimodal data")
    parser.add_argument("--test_path", type=str, default="test_0_10.csv", help="the path of multimodal data")
    parser.add_argument("--prompt_type", type=str, default="1", help="the type of prompt")
    parser.add_argument("--use_context", action="store_true", help="whether use context for ScienceQA")
    parser.add_argument("--max_output_new_length", type=int, help="the length of max new generative tokens")
    ##for lavin
    parser.add_argument("--ckpt_dir", type=str, default="/datas/multimodal_LLMs/LLaMA-7B", help="the path of checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="/datas/multimodal_LLMs/LLaMA-7B/tokenizer.model", help="the path of tokenizer")
    parser.add_argument("--adapter_path", type=str, default='/data/xiaocui/code/Multimodal_LLMs/LaVIN/weights/sqa-llama-7b.pth', help="the path of tokenizer")
    parser.add_argument("--temperature", type=float, default=0.8, help="")
    parser.add_argument("--generation_temperature", type=float, default=0.1, help="")
    parser.add_argument("--n_prompt", type=int, default=6, help="")
    parser.add_argument("--top_p", type=float, default=0.75, help="")
    parser.add_argument("--max_seq_len", type=int, default=512, help="")
    parser.add_argument("--max_gen_len", type=int, default=128, help="")
    parser.add_argument("--max_batch_size", type=int, default=1, help="")
    parser.add_argument("--local_rank", type=str, default="1", help="")
    parser.add_argument("--llm_model", type=str, default="LLaMA-13B", help="")
    parser.add_argument("--visual_adapter_type", type=str, default="router", help="")
    parser.add_argument("--adapter_type", type=str, default="repattn", help="")
    parser.add_argument("--bits", type=str, default="16bits", help="")
    ##for lynx_llm
    parser.add_argument("--llama_path", type=str, default="/data/xiaocui/weights/decapoda-research/llama-7b-hf", help="the path of llama-7b")
    ## ''decapoda-llama-7b-hf' --llama_path "/data/xiaocui/weights/decapoda-research/llama-7b-hf"
    ## ''decapoda-llama-13b-hf' --llama_path "/data/xiaocui/weights/decapoda-research/llama-13b-hf"
    parser.add_argument("--open_flamingo_path", type=str, default="/data/xiaocui/weights/openflamingo/OpenFlamingo-9B/checkpoint.pt", help="the path of openflamingo")
    parser.add_argument("--finetune_path", type=str, default="/data/xiaocui/weights/Multimodal_GPT/mmgpt-lora-v0-release.pt", help="the path of mmgpt_lora")
    
    ## for llama
    parser.add_argument('--use_quantization', action='store_true', help='use quantization or not in llama')
    parser.add_argument('--peft_model', type=str, default=None, help='the path of peft_model in llama')
    parser.add_argument("--top_k", type=int, default=1, help="")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="")
    parser.add_argument("--length_penalty", type=int, default=1, help="")
    
    ##for mplug_owl
    parser.add_argument('--mplug_owl_pretrained_ckpt', type=str, default='/data/xiaocui/weights/mplug/MAGAer13/mplug-owl-llama-7b', help='the path of the pretrained weights for mplug_owl')
    
    ##for minigpt4
    parser.add_argument("--cfg_path", type=str, default="multimodal_eval_main/models/MiniGPT4/eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument( "--options", help="override some settings in the used config, the key-value pair " 
                                                        "in xxx=yyy format will be merged into config file (deprecate), "
                                                        "change to --cfg-options instead.",)
    parser.add_argument("--minigpt4_pretrained_ckpt", type=str, default="/data/xiaocui/weights/MiniGPT4/pretrained_minigpt4.pth", help="path to minigpt4 pretrained weights.")
    
    
    ## for llama_adapter
    parser.add_argument("--llama_path_for_llama_adapter", type=str, default="/data/xiaocui/weights/LLaMA-7B", help="the path of llama-7b")
    
    ## for vpgtrans
    # parser.add_argument("--cfg_path", type=str, default="multimodal_eval_main/models/VPGTrans/lavis/projects/blip2/demo/vl_vicuna_demo.yaml", help="path to configuration file.")
    
    ## for llava
    parser.add_argument("--llava_model_path", type=str, default="/data/xiaocui/weights/llava-7b", help="/path/to/model")
    parser.add_argument("--conv_mode", type=str, default='multimodal')
    
    ## for chatgpt
    parser.add_argument("--chatgpt_engine", type=str, default="", help="the engine for chatgpt")
    parser.add_argument("--api_key", type=str, default="", help="your API Key for chatgpt")
    
    return parser.parse_args()
args = parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

##for LaVIN
os.environ["LOCAL_RANK"]=args.local_rank


def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


@retry(wait=wait_fixed(5), stop=stop_after_attempt(6), before=before_retry_fn)
def query_chatgpt_model(api_key: str, engine: str, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 256, temperature: float = 0):
    openai.api_type = "azure"
    openai.api_base = "https://research2.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key =api_key
    try:
        completions = openai.ChatCompletion.create(
            engine=engine,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        output = completions.choices[0].message.content.strip()
    except Exception as e:
        print(e)
    return output

def parallel_query_chatgpt_model(api_key, engine, prompt):
    return query_chatgpt_model(api_key, engine, prompt)

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    dist.init_process_group(backend='nccl', init_method='env://')
    print('+++++++++++++++++++++local_rank is {}+++++++++++'.format(local_rank))
    print('+++++++++++++++++++++WORLD_SIZE is {}+++++++++++'.format(world_size))
    # torch.distributed.init_process_group("nccl")
    LaVIN_initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(42)
    return local_rank, world_size

def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = LaVIN_Tokenizer(model_path=str(Path(llama_model_path) / model_name / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B':
        checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()
    if mp_world_size == len(checkpoints):
        print('same number of shards of checkpoints and training, loading directly...')
        dist.barrier()
        print('[rank=%d, mp_rank=%d] loading from %s' % (dist.get_rank(), mp_rank, checkpoints[mp_rank]))
        checkpoint = torch.load(checkpoints[mp_rank], map_location='cpu')
    else:
        print('different number of shards of checkpoints and training, redistributing...')
        if dist.get_rank() == 0:
            loaded = []
            for x in checkpoints:
                print('loading from', x)
                loaded.append(torch.load(x, map_location='cpu'))

            full_state_dict = {}
            split_dims = {}

            def add_weight_with_split_dim(name, dim):
                if dim < 0:  # bcast without split
                    full_state_dict[name] = loaded[0][name].clone()
                else:
                    full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
                for x in loaded:
                    del x[name]
                split_dims[name] = dim

            add_weight_with_split_dim('tok_embeddings.weight', 1)
            add_weight_with_split_dim('norm.weight', -1)
            add_weight_with_split_dim('output.weight', 0)
            for i in range(params['n_layers']):
                print('gathering layer %d of %d' % (i, params['n_layers']))
                layer_prefix = f'layers.{i}.'
                bcast_names = [
                    'attention_norm.weight',
                    'ffn_norm.weight',
                ]
                column_parallel_names = [
                    'attention.wq.weight',
                    'attention.wk.weight',
                    'attention.wv.weight',
                    'feed_forward.w1.weight',
                    'feed_forward.w3.weight',
                ]
                row_parallel_names = [
                    'attention.wo.weight',
                    'feed_forward.w2.weight',
                ]
                for key in bcast_names:
                    add_weight_with_split_dim(layer_prefix + key, -1)
                for key in column_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 0)
                for key in row_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 1)

            full_state_dict_meta = dict((k, v.shape) for k, v in full_state_dict.items())
            dist.broadcast_object_list([full_state_dict_meta, split_dims], src=0)

        else:  # dist.get_rank() != 0
            recv_objs = [None, None]
            dist.broadcast_object_list(recv_objs, src=0)
            full_state_dict_meta, split_dims = recv_objs

        local_state_dict = {}
        for k in sorted(full_state_dict_meta.keys()):
            print('redistributing weights: %s' % k)
            if dist.get_rank() == 0:
                value = full_state_dict[k].cuda().half()
                del full_state_dict[k]
            else:
                value = torch.empty(full_state_dict_meta[k], device='cuda', dtype=torch.half)
            dist.broadcast(value, src=0)
            value = value.cpu()
            if split_dims[k] < 0:
                local_state_dict[k] = value
            else:
                dim = split_dims[k]
                assert dim >= 0 and dim < value.ndim and value.size(dim) % mp_world_size == 0
                shard_size = value.size(dim) // mp_world_size
                shard_st, shard_ed = shard_size * mp_rank, shard_size * (mp_rank + 1)
                # TODO: make more general
                if dim == 0:
                    value = value[shard_st: shard_ed]
                elif dim == 1:
                    value = value[:, shard_st: shard_ed]
                else:
                    raise NotImplementedError()
                local_state_dict[k] = value.clone()

        checkpoint = local_state_dict

    return checkpoint, tokenizer, params

def load_LaVIN_model(ckpt_dir: str='/data/xiaocui/weights',
    llm_model:str='7B',
    tokenizer_path: str='/datas/multimodal_LLMs/LLaMA-7B/tokenizer.model',
    adapter_path: str='/data/xiaocui/weights/LaVIN/LaIN-7B/weights/sqa-llama-7b.pth',
    local_rank: int=0,
    world_size: int=1,
    max_seq_len: int=128,
    max_batch_size: int=1,
    adapter_type: str='repattn',
    adapter_dim:int=8,
    adapter_scale:float=1,
    hidden_proj:int=128,
    visual_adapter_type: str='router',
    temperature: float=0.8,
    use_vicuna: bool=False,
    bits: str='16bits',
    cpu_load:bool=False,
) -> LaVIN_Generator:
    start_time = time.time()
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(ckpt_dir, llm_model)

    print("Loading")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")


    model_args: LaVIN_ModelArgs = LaVIN_ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,hidden_proj=hidden_proj, **params
    )
    model_args.vocab_size = tokenizer.n_words
    
    if cpu_load:
        #cpu load is slow, but is freindly for GPU with limited memory.
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    model = LaVIN_Transformer(model_args)
    #delete language encoder
    del model.backbone.transformer

    torch.set_default_tensor_type(torch.FloatTensor)

    if bits in ['4bit','8bit']:
        from multimodal_eval_main.models.LaVIN.util.quantization import quant_model_bnb
        model.layers = quant_model_bnb(model.layers, quant_bit='4bit')
        
    LaVIN_set_MMAdapter(model, adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)
    LaVIN_set_Clip_Adapter(model.backbone.visual, visual_adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)

    model.load_state_dict(checkpoint, strict=False)

    if use_vicuna:
        apply_model_delta_online(model,'../data/weights/vicuna_'+llm_model)

    state_dict={}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.','')]=adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))
    # parameters = get_parameter_number(model)
    # print("+++++++++++++++++++++++++++++++++++++++++++=") 
    # print(parameters)

    generator = LaVIN_Generator(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator



def load_model(args, loacal_rank=None, world_size=None):
    '''
    name="blip2_t5", model_type="pretrain_flant5xxl"
    '''
    print(f"++++++++++++++++Loading model is {args.model_name}+++++++")
    ###Flan-T5
    if args.model_name == 'text_flan-t5-xxl':
        model_type = 'google/flan-t5-xxl'
        tokenizer = T5Tokenizer.from_pretrained(model_type)
        model = T5ForConditionalGeneration.from_pretrained(model_type, device_map="auto")
        args.model_type = model_type
        return tokenizer, model, model_type
    
    ##LLaMA-V1
    elif 'decapoda-llama'in args.model_name or 'meta-llama2' in args.model_name:
        if args.model_name == 'decapoda-llama-7b-hf':
            ## llama_path = 'decapoda-research/llama-7b-hf' or 'your local path'
            model_type = 'LLaMA-V1-7B'
        elif args.model_name == 'decapoda-llama-13b-hf':
            model_type = 'LLaMA-V1-13B'
            ## llama_path = 'decapoda-research/llama-13b-hf' or 'your local path'
        elif args.model_name == 'meta-llama2-7b-hf':
            model_type = 'LLaMA-V2-7B'
            ## llama_path = 'meta-llama/Llama-2-7b-hf' or 'your local path' 
        elif args.model_name == 'meta-llama2-13b-hf':
            model_type = 'LLaMA-V2-13B'
            ## llama_path = 'meta-llama/Llama-2-13b-hf' or 'your local path' 
            
            
        model = load_model_for_llama(args.llama_path, args.use_quantization)
        tokenizer = LlamaTokenizer.from_pretrained(args.llama_path)
        tokenizer.add_special_tokens(
                                            {
                                                "pad_token": "[PAD]",
                                            }
                                        )
        if args.peft_model:
            model = load_peft_model_for_llama(model, args.peft_model)     
        model.eval() 
        args.model_type = model_type
        return tokenizer, model, model_type
    
    ## BLIP2   
    elif args.model_name == 'blip2_t5':
        model_type='blip2-flan-t5-xxl'
        model = BlipModel(path_model="/data/xiaocui/weights/Salesforce/{}".format(model_type), max_output_length=args.max_output_new_length)
        args.model_type = model_type
        return model, model_type
    
    ###InstructBLIP
    elif args.model_name == 'blip2_instruct_flant5xxl':
        model_type="flant5xxl"
        model = InstructBlipT5Model(path_model=model_type, max_output_length=args.max_output_new_length)
        args.model_type = model_type
        return model, model_type
    
    ##Fromage
    elif args.model_name == 'fromage':
        model_type="Fromage-9B"
        model = FromageModel(model_type=model_type, max_output_length=args.max_output_new_length)
        args.model_type = model_type
        return model, model_type
    
    ##OpenFlamingo
    elif args.model_name == 'openflamingo':
        model_type='OpenFlamingo-9B'
        model = OpenFlamingoModel(model_type=model_type, max_output_length=args.max_output_new_length)
        args.model_type = model_type
        return model, model_type
    ##MultimodalGPT
    elif args.model_name == 'mmgpt':
        model_type = 'mmgpt_lora_v0_release_9B'
        model = mmgpt_Inferencer(
            llama_path=args.llama_path,
            open_flamingo_path=args.open_flamingo_path,
            finetune_path=args.finetune_path) 
        args.model_type = model_type
        return model, model_type
    
    ###LaVIN
    elif  'LaVIN' in args.model_name:
        if args.model_name == 'LaVIN_7B':
            model_type = '7B'
        elif args.model_name == 'LaVIN_13B':
            model_type = '13B'
        model = load_LaVIN_model(ckpt_dir=args.ckpt_dir,
                                llm_model=args.llm_model,
                                tokenizer_path=args.tokenizer_path,
                                adapter_path=args.adapter_path,
                                local_rank=loacal_rank,
                                world_size=world_size,
                                max_seq_len=args.max_seq_len,
                                max_batch_size=args.max_batch_size,
                                adapter_type=args.adapter_type,
                                bits=args.bits,
                                visual_adapter_type=args.visual_adapter_type,
                                temperature=args.temperature,
                                )
        args.model_type = model_type
        return model, model_type
    
    ###mPLUG-Owl
    elif args.model_name == 'mplug_owl':
        model_type = 'mplug_owl_llama_7b'
        model = MplugOwlForConditionalGeneration.from_pretrained(
                                                                    args.mplug_owl_pretrained_ckpt,
                                                                    torch_dtype=torch.bfloat16,
                                                                ).to(device)
        model.tie_weights()
        image_processor = MplugOwlImageProcessor.from_pretrained(args.mplug_owl_pretrained_ckpt)
        tokenizer = LlamaTokenizer.from_pretrained(args.mplug_owl_pretrained_ckpt)
        processor = MplugOwlProcessor(image_processor, tokenizer)
        args.model_type = model_type
        return tokenizer, model, processor, model_type
    
    ### MiniGPT4
    elif args.model_name == 'minigpt4':
        model_type = 'MiniGPT4_Vicuna13B'
        args.model_type = model_type
        cfg = minigpt4_Config(args)
        model_config = cfg.model_cfg
        model_config.ckpt = args.minigpt4_pretrained_ckpt
        # model_config.device_8bit = args.gpu_id
        model_cls = miniGPT4_registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = miniGPT4_registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        return model, vis_processor, model_type
    
    ## LLaMA_Adapterv2
    elif args.model_name == 'llama_adapterv2':  
        model_type = 'LLaMA_AdapterV2_7B' 
        args.model_type = model_type
        model, vis_processor = llama_adapter.load("BIAS-7B", args.llama_path_for_llama_adapter, device)
        return model, vis_processor, model_type

    ## VPGTrans
    elif args.model_name == 'vpgtrans':
        
        model_type = 'VPGTrans_Vicuna7B'
        args.model_type = model_type
        cfg = VPGTans_Config(args)
        model_config = cfg.model_cfg
        model_cls = vpgtrans_registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)
        
        vis_processor_cfg = cfg.datasets_cfg.minigpt4_self_instruct_caption.vis_processor.train
        print(f'=========================the vis_processor_cfg is {vpgtrans_registry.get_processor_class(vis_processor_cfg.name)}')
        vis_processor = vpgtrans_registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        return model, vis_processor, model_type
    
    ## LLaVA
    elif 'llava' in args.model_name:
        if args.model_name == 'llava_7b':
            model_type = 'llava_7b'
        elif args.model_name == 'llava_13b':
            model_type = 'llava_13b'
        args.model_type = model_type
        tokenizer = AutoTokenizer.from_pretrained(args.llava_model_path)
        if "mpt" in args.llava_model_path.lower():
            model = LlavaMPTForCausalLM.from_pretrained(args.llava_model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                        use_cache=True).to(device)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(args.llava_model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                        use_cache=True).to(device)
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

        return tokenizer, model, image_processor, model_type, mm_use_im_start_end
    else:
        print("++++++++++++++++++++++++You can add the other large language models that you want to evaluated!!!! ++++++++++++++++++++++++++++++++++++++++++++")

# Get label space
def get_label_space(task: str, dataset: str) -> list:
    if task == 'MABSA':
        if dataset == 'Twitter_2015' or dataset == "Twitter_2017":
           label_space = ["positive", "neutral", "negative"]
        elif dataset == 'MASAD':
            label_space = ["positive", "negative"] 
    elif task == 'MSA':
        if dataset == 'MVSA_Multiple' or dataset == 'MVSA_Single' or dataset == "MOSI_3":
            label_space = ["positive", "neutral", "negative"]
        elif dataset == "MOSI_2" or dataset == "MOSEI_2":
            label_space = ["positive", "negative"]
        elif "MOSI_7" in dataset or dataset == "MOSEI_7":
            label_space = ["strongly positive", "positive", "weakly positive", "neutral", "weakly negative", "negative", "strongly negative"]
        elif dataset == 'TumEmo':
            label_space = ["angry", "bored", "calm", "fear", "happy", "love", "sad"]
    elif task == "MRE":
        '''
       {'held_on': 18, 'couple': 19, 'member_of': 110, 'alternate_names': 29, 'peer': 156, 'contain': 99, 'nationality': 10, 'subsidiary': 16, 'part_of': 14, 'locate_at': 46, 'place_of_birth': 7, 'present_in': 74, 'charges': 1, 'parent': 4, 'place_of_residence': 29, 'awarded': 4, 'siblings': 1, 'religion': 1, 'neighbor': 2})
        '''
        entity_cat_space = ['loction', 'organization', 'person', 'misc']
        label_space = ['held_on', 'couple', 'member_of', 'alternate_names', 'peer', 'contain', 'nationality', 'subsidiary', 'part_of', 'locate_at', 'place_of_birth', 'present_in', 'charges', 'parent', 'place_of_residence', 'awarded', 'siblings', 'religion', 'neighbor']
        
        if "JMNRE" in dataset:
            label_space = (sorted(label_space), sorted(entity_cat_space))
        return label_space
    elif task== "MHM":
        ##Multimodal_Hateful_Memes
        label_space = ["yes", "no"]
    elif task=="MSR":
        label_space = ["yes", "no"]
    elif task=="QA":
        if dataset == "ScienceQA" or dataset == "ScienceQA_no_image" or dataset=="ScienceQA_1":
            label_space = ["0", '1', "2", "3", "4"]
    else:
        raise NotImplementedError
    return sorted(label_space)


# Function to get the task name and stance target based on the task and dataset
def get_task_name(task: str, dataset: str) -> str:

    if task == 'MABSA':
        if dataset == 'Twitter_2015' or dataset == "Twitter_2017" or dataset == 'MASAD' :
          task_name = "multimodal aspect-based sentiment classification"
    elif task == 'MSA':
        if dataset == 'MVSA_Multiple' or dataset == 'MVSA_Single' or dataset == 'TumEmo' or dataset == "MOSI_3" or ("MOSI_7" in dataset) or dataset == "MOSI_2" or dataset == "MOSEI_2" or dataset == "MOSEI_7":
          task_name = "multimodal sentiment classification"
    elif task == "MRE":
        if 'JMNRE' in dataset:
            task_name = "joint multimodal entity-relation extraction"
        elif dataset == "MNRE":
            task_name = "multimodal relation extraction"
    elif task == "MHMR":
        task_name = "multimodal hateful detection"
    elif task == "MSR":
        task_name = "multimodal irony detection" 
    elif task == "QA":
        task_name ="multimodal question answer"   
    else:
        raise NotImplementedError

    return task_name.title()


def generate_fake_data(task, dataset, label_space, row):
    # fake data for dev
    if any(substring in dataset for substring in ["uabsa", "aste", "asqp"]):
        try:
            pred = [random.choice(eval(row["label_text"]))]
        except:
            pred = []
    else:
        pred = str(random.choice(label_space))
    return pred

# Define templates for different tasks and datasets
def generate_multimodal_template(key, label_space, task_name, **kwargs):
    task_definitions = {
        "MABSA": "Given the text-image pair and the aspect, assign a sentiment label towards \"{target}\" from {label_space}.",
        "MSA": "Given the text-image pair, assign a sentiment label from {label_space}.",
        "MNRE": "Given the text-image pair, assign a relation label towards the head entity \"{head_entity}\" belongs to \"{head_cat}\" and the tail entity \"{tail_entity}\" belongs to \"{tail_cat}\" from {label_space}.",
        'MHM': "Given the text-image pair, please determine whether or not it contains hate. Assign a label from {label_space}.",
        "MSR": "Given the text-image pair, please determine whether or not it contains irony. Assign a label from {label_space}.",
        "QA": "Given the question, "
    }

    output_formats = {
        "MABSA": "Return label only without any other text.",
        "MSA": "Return label only without any other text.",
        "MNRE": "Return label only without any other text.",
        "MHM": "Return label only without any other text.",
        "MSR": "Return label only without any other text.",
        "QA": "Return answer only without any other text.",
    }
    
    

    if key == "stance":
        task_name += " ({target})".format(**kwargs)

    task_definition = task_definitions[key].format(**kwargs, label_space=label_space)
    output_format = output_formats[key]

    return task_name, task_definition, output_format


def generate_text_template(key, label_space, task_name, **kwargs):
    task_definitions = {
        "MABSA": "Given the text and the aspect, assign a sentiment label towards \"{target}\" from {label_space}.",
        "MSA": "Given the text, assign a sentiment label from {label_space}.",
        "MRE": "Given the text, assign a relation label towards the head entity \"{head_entity}\" belongs to \"{head_cat}\" and the tail entity \"{tail_entity}\" belongs to \"{tail_cat}\" from {label_space}.",
        'MHMR': "Given the text, please determine whether or not it contains hate. Assign a label from {label_space}.",
        "MSR": "Given the text, please determine whether or not it contains irony. Assign a label from {label_space}.",
        "QA": "Given the question, "
    }

    output_formats = {
        "MABSA": "Return label only without any other text.",
        "MSA": "Return label only without any other text.",
        "MRE": "Return label only without any other text.",
        "MHMR": "Return label only without any other text.",
        "MSR": "Return label only without any other text.",
        "QA": "Return answer only without any other text.",
    }
    
    

    if key == "stance":
        task_name += " ({target})".format(**kwargs)

    task_definition = task_definitions[key].format(**kwargs, label_space=label_space)
    output_format = output_formats[key]

    return task_name, task_definition, output_format


# generate demos
def generate_fix_demo(train_df, task, dataset):
    tuple_list = []
    if dataset in ['Twitter_2015', "Twitter_2017", 'MASAD']:
        for i, row in train_df.iterrows():
            aspect = row["aspect"]
            text = row["text"].replace('$T$', aspect)
            label = row["label_text"]
            text += f" (sentiment towards Aspect: \"{aspect}\")"
            image_path = row['image']
            image_description = row['image_description']
            tuple_list.append((text, label, image_path, image_description))
    elif dataset in ['MVSA_Single', 'MVSA_Multiple', 'TumEmo', 'MOSI_3', "MOSI_7", "MOSI_2", "MOSI_7_1", "MOSEI_2", 'MOSEI_7']:
        for i, row in train_df.iterrows():
            text = row["text"]
            image_path = row['image']
            label = row["label_text"]
            image_description = row['image_description']
            tuple_list.append((text, label, image_path, image_description))
    elif dataset in ['hate', 'MSD']:
        for i, row in train_df.iterrows():
            text = row["text"]
            image_path = row['image']
            label = row["label_text"]
            image_description = row['image_description']
            tuple_list.append((text, label, image_path, image_description))     
    elif dataset == "MNRE":
        for i, row in train_df.iterrows():
            text = row["text"]
            head_entity = row['head_entity']
            head_cat = row['head_cat']
            tail_entity = row['tail_entity']
            tail_cat = row['tail_cat']
            label = row["label_text"]
            text += f" (relation towards the head entity \"{head_entity}\" belongs to \"{head_cat}\" and the tail entity \"{tail_entity}\" belongs to \"{tail_cat}\")"
            image_path = row['image']
            image_description = row['image_description']
            tuple_list.append((text, label, image_path, image_description))        
    else:
        sub_df = train_df[['text', 'label_text']]
        tuple_list = [tuple(x) for x in sub_df.to_records(index=False)]
    return tuple_list


# Function to generate prompt for the OpenAI model
def generate_prompt(setting, task, dataset, label_space, row, demo_tuples, model_name, prompt_type, args):
    
    if task!="QA":
        text = row["text"]
        if task == 'MABSA':
            aspect = row['aspect']
        task_name = get_task_name(task, dataset)

        if task == "MABSA":
            if model_name == 'text_flan-t5-xxl' or 'decapoda-llama'in args.model_name or model_name == 'chatgpt':
                task_name, task_definition, output_format = generate_text_template("MABSA", label_space, task_name=task_name, target=row["aspect"])
            else:
                task_name, task_definition, output_format = generate_multimodal_template("MABSA", label_space, task_name=task_name, target=row["aspect"])
        elif task == "MSA":
            if model_name == 'text_flan-t5-xxl' or 'decapoda-llama'in args.model_name or model_name == 'chatgpt':
                task_name, task_definition, output_format = generate_text_template("MSA", label_space, task_name=task_name)
            else:
                task_name, task_definition, output_format = generate_multimodal_template("MSA", label_space, task_name=task_name)
        elif task=='MRE':
            head_entity = row['head_entity']
            head_cat = row['head_cat']
            tail_entity = row['tail_entity']
            tail_cat = row['tail_cat']
            if dataset == "MRE":
                relation_label_space = label_space
                if model_name == 'text_flan-t5-xxl' or 'decapoda-llama'in args.model_name or model_name == 'chatgpt':
                    task_name, task_definition, output_format = generate_text_template("MRE", relation_label_space, task_name=task_name, head_entity=head_entity, head_cat=head_cat, tail_entity=tail_entity, tail_cat=tail_cat)
                else:
                    task_name, task_definition, output_format = generate_multimodal_template("MRE", relation_label_space, task_name=task_name, head_entity=head_entity, head_cat=head_cat, tail_entity=tail_entity, tail_cat=tail_cat) 
        elif task == "MHMR":
            if model_name == 'text_flan-t5-xxl' or 'decapoda-llama'in args.model_name or model_name == 'chatgpt':
                task_name, task_definition, output_format = generate_text_template("MHMR", label_space, task_name=task_name)
            else:
                task_name, task_definition, output_format = generate_multimodal_template("MHMR", label_space, task_name=task_name)
        elif task == "MSR":
            if model_name == 'text_flan-t5-xxl' or 'decapoda-llama'in args.model_name or model_name == 'chatgpt':
                task_name, task_definition, output_format = generate_text_template("MSR", label_space, task_name=task_name)
            else:
                task_name, task_definition, output_format = generate_multimodal_template("MSR", label_space, task_name=task_name)
        else:
            raise NotImplementedError
    else:
        ##original_index,question,image,answer,answer_text,choices,hint
        text = row['question']
        choices = eval(row['choices'])
        task_name = get_task_name(task, dataset)
        if model_name == 'text_flan-t5-xxl'  or 'decapoda-llama'in args.model_name or model_name == 'chatgpt':
            task_name, task_definition, output_format = generate_text_template("QA", label_space='', task_name=task_name)
        else:
            task_name, task_definition, output_format = generate_multimodal_template("QA", label_space='', task_name=task_name)
        option_num = ["(a)", "(b)", "(c)", "(d)", "(e)","(f)", "(g)", "(h)" ]
       

        options =''       
        # question = "What is the answer about the above question?"
        for i, choice in enumerate(choices):
            option = option_num[i]+ " " + choice + " "
            options +=option
        task_definition  = task_definition + f"please choose the answer from \"{options}\" to the following question."
        
    if setting == "zero-shot":
        if prompt_type == "1":
            if task=="MABSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\nLabel:"
                question = ""
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                prompt = prompt+"Label:"
            elif task=='QA':
                context = row['hint']
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\nLabel:"
                        question = "" 
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nLabel:"
                        question = ""
            else:
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\nLabel:"
        
        elif prompt_type == "2":
            if task=="MABSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
            elif task == "MSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity based on the text-image pair?\n"
            elif task == "MHMR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hate?\n"
            elif task == "MSR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or notthe text-image pair contains irony?\n"
            elif task=='QA':
                context = row['hint']
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
            if task=="QA":   
                prompt = prompt + "The answer is:"
            else:
                prompt = prompt + "Question: " + question + "Answer:"
            
        elif prompt_type == "3":
            task_predefinition = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
            if task=="MABSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset == 'MASAD':
                    options = "(a) negative (b) positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
            elif task == "MSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset =="TumEmo":
                    options = "(a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad"
                elif dataset == "MOSI_2" or dataset == "MOSEI_2":
                    options = "(a) negative (b) positive"
                elif "MOSI_7" in dataset or dataset == "MOSEI_7":
                    # options = "(a) strongly positive (b) positive (c) weakly positive (d) neutral (e) weakly negative (f) negative (g) strongly negative"
                    options = "(a) negative (b) neutral  (c) positive (d) strongly negative (e) strongly positive (f) weakly negative (g) weakly positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
                    
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity based on the text-image pair?\n"
                options = "(a) held_on (b) couple (c) member_of (d) alternate_names (e) peer (f) contain (g) nationality (h) subsidiary (i) part_of (j) locate_at (k) place_of_birth (l) present_in (m) charges (n) parent (o) place_of_residence (p)awarded (q) siblings (r) religion (s) neighbor'"
            elif task == "MHMR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hate?\n"
                options = "(a) yes (b) no"
            elif task == "MSR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "(a) yes (b) no"
            elif task=='QA':
                context = row['hint']
                
                if args.use_context:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\n Question: {text}\nContext: {context}\n"
                else:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\n Question: {text}\n"
                  
            if task=="QA":        
                prompt = task_predefinition + "### Instruction: \n" + prompt + f"Options: {options}\n" + "### Response:"
            else:
                prompt = task_predefinition + "### Instruction: \n" + prompt + "### Instruction: \n" + question + f"Options: {options}\n" + "### Response:"
                
        elif prompt_type == "4":
            if task=="MABSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
            elif task == "MSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity based on the text-image pair?\n"
            elif task == "MHMR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hate?\n"
            elif task == "MSR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            elif task=='QA':
                context = row['hint']
                if args.use_context:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                    question = "What is the answer about the above question?"
                else:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
                    question = "What is the answer about the above question?"
                          
            prompt = prompt + question 
            
        elif prompt_type == "5":
            if task=="MABSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset == 'MASAD':
                    options = "(a) negative (b) positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
            elif task == "MSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset =="TumEmo":
                    options = "(a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad"
                elif dataset == "MOSI_2" or dataset == "MOSEI_2":
                    options = "(a) negative (b) positive"
                elif "MOSI_7" in dataset or dataset == "MOSEI_7":
                    # options = "(a) strongly positive (b) positive (c) weakly positive (d) neutral (e) weakly negative (f) negative (g) strongly negative"
                    options = "(a) negative (b) neutral  (c) positive (d) strongly negative (e) strongly positive (f) weakly negative (g) weakly positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
                    
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity based on the text-image pair?\n"
                options = "(a) held_on (b) couple (c) member_of (d) alternate_names (e) peer (f) contain (g) nationality (h) subsidiary (i) part_of (j) locate_at (k) place_of_birth (l) present_in (m) charges (n) parent (o) place_of_residence (p)awarded (q) siblings (r) religion (s) neighbor'"
            elif task == "MHMR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hate?\n"
                options = "(a) yes (b) no"
            elif task == "MSR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "(a) yes (b) no"
            elif task=='QA':
                context = row['hint']
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
                    # question = "What is the answer about the above question?"
            if task=="QA":
                prompt = prompt + f"Options: {options}\n" + "The answer is:"
            else:
                prompt = prompt + "Question: " + question  + f"Options: {options}\n" + "Answer:"
            
        elif prompt_type == "6":
            task_predefinition = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            
            if task == "MSA":        
                prompt = f"Please perform {task_name} task. {task_definition} {output_format}\nHuman: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                # prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
            elif task=="MABSA":   
                prompt = f"Please perform {task_name} task. {task_definition} {output_format}\nHuman: {text} Aspect: {aspect} \n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                # prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task. {task_definition} {output_format}\nHuman: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity based on the text-image pair?\n"
            elif task == "MHMR":
                prompt = f"Please perform {task_name} task. {task_definition} {output_format}\nHuman: {text}\n"
                question = "whether or not the text-image pair contains the hate?\n"
            elif task == "MSR":
                prompt = f"Please perform {task_name} task. {task_definition} {output_format}\nHuman: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
            
            elif task=='QA':
                context = row['hint']
                if args.use_context:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                else:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
            if task == "QA":
                prompt = task_predefinition + "Human: " + prompt + "AI:"
            else:
                prompt = task_predefinition + "Human: " + prompt + "Human: " + question + "AI:"
                
        elif prompt_type == "7":
            if task == "MSA":        
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset == 'MVSA_Single' or dataset == 'MVSA_Multiple' or dataset == "MOSI_3" :
                    options = "neutral or negative or positive \n"
                elif dataset == "MOSI_2" or dataset == "MOSEI_2":
                    options = "negative or positive \n"
                elif "MOSI_7" in dataset or dataset == "MOSEI_7":
                    options = "strongly positive or positive or weakly positive or neutral or weakly negative or negative or strongly negative\n"
                else:
                    options = "angry or bored or calm or fear or happy or love or sad \n"
                # prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
            elif task=="MABSA":   
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect} \n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset!= 'MASAD':
                    options = 'neutral or negative or positive \n'
                else:
                    options = 'negative or positive \n'
                # prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity based on the text-image pair?\n"
                options =  "held_on or couple or member_of or alternate_names or peer or contain or nationality or subsidiary or part_of or locate_at or place_of_birth or present_in or charges or parent or place_of_residence or awarded or siblings or religion or neighbor\n"
            elif task == "MHMR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains the hate?\n"
                options = "yes or no \n"
            elif task == "MSR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "yes or no \n"
            elif task=='QA':
                context = row['hint']
                choices = eval(row['choices'])
                option_num = ["(a)", "(b)", "(c)", "(d)", "(e)","(f)", "(g)", "(h)" ]
                new_options =''
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    if args.use_context:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                    else:
                        prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
                    for i, choice in enumerate(choices):
                        choice = "\"" + choice + "\""
                        if i < (len(choices)-1):
                            option = choice + " or "
                        else:
                            option = choice
                        new_options +=option
            if task =="QA":
                prompt = prompt + "Options: " +  new_options  + "\nThe answer is:"
            else:
                prompt = prompt + "Question: " + question + "Options: " + options + "Answer:"
                
        elif prompt_type == "8":
            if task=="MABSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} Aspect: {aspect}\n"
            elif task == "MSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
            elif task == "MHMR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task == "MSR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text}\n"
            elif task=='QA':
                context = row['hint']
                
                if args.use_context:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                else:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
        
            prompt = prompt       
        
        elif prompt_type == "9":
            task_predefinition = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n"
            if task=="MABSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input:\n {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset == 'MASAD':
                    options = "(a) negative (b) positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
            elif task == "MSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input:\n {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset !="TumEmo":
                    options = "(a) neutral (b) negative (c) positive"
                else:
                    options = "(a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad"
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity based on the text-image pair?\n"
                options = "(a) held_on (b) couple (c) member_of (d) alternate_names (e) peer (f) contain (g) nationality (h) subsidiary (i) part_of (j) locate_at (k) place_of_birth (l) present_in (m) charges (n) parent (o) place_of_residence (p)awarded (q) siblings (r) religion (s) neighbor'"
            elif task == "MHMR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input:\n {text}\n"
                question = "whether or not the text-image pair contains the hate?\n"
                options = "(a) yes (b) no"
            elif task == "MSR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n### Input:\n {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "(a) yes (b) no"
            elif task=='QA':
                context = row['hint']
                
                prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\n"
                
            if task =="QA":
                if args.use_context:
                    prompt = task_predefinition + "### Instruction: \n" + prompt + "### Input: \n" + f"Question: {text}\nContext: {context}\n" + "### Response:"
                else:
                    prompt = task_predefinition + "### Instruction: \n" + prompt + "### Input: \n" + f"Question: {text}\n" + "### Response:"
          
            else:
                prompt = task_predefinition + "### Instruction: \n" + prompt + "### Input: \n" + question + "### Response:"
    
        elif prompt_type == "10":
            if task=="MABSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSequence: {text} Aspect: {aspect}\n"
                question = "what is the sentiment about the aspect based on the text-image pair?\n"
                if dataset == 'MASAD':
                    options = "(a) negative (b) positive"
                else:
                    options = "(a) neutral (b) negative (c) positive"
            elif task == "MSA":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSequence: {text}\n"
                question = "what is the sentiment about the text-image pair?\n"
                if dataset !="TumEmo":
                    options = "(a) neutral (b) negative (c) positive"
                else:
                    options = "(a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad"
            elif task == "MRE":
                head_entity = row['head_entity']
                head_cat = row['head_cat']
                tail_entity = row['tail_entity']
                tail_cat = row['tail_cat']
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}.\n"
                question = "what has relation between the head entity and the tail entity based on the text-image pair?\n"
                options = "(a) held_on (b) couple (c) member_of (d) alternate_names (e) peer (f) contain (g) nationality (h) subsidiary (i) part_of (j) locate_at (k) place_of_birth (l) present_in (m) charges (n) parent (o) place_of_residence (p)awarded (q) siblings (r) religion (s) neighbor'"
            elif task == "MHMR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSequence: {text}\n"
                question = "whether or not the text-image pair contains the hate?\n"
                options = "(a) yes (b) no"
            elif task == "MSR":
                prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nSequence: {text}\n"
                question = "whether or not the text-image pair contains irony?\n"
                options = "(a) yes (b) no"
            elif task=='QA':
                context = row['hint']
                if args.use_context:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\nContext: {context}\n"
                else:
                    prompt = f"Please perform {task_name} task.\n{task_definition} {output_format}\nQuestion: {text}\n"
                
            if task == "QA":
                prompt = "User: " + prompt + ":<answer>"
            else:
                prompt = "User: " + prompt + "Question: " + question + ":<answer>"
            
        elif prompt_type == "11":
            if task=='QA':
                context = row['hint']
                choices = eval(row['choices'])
                option_num = ["(a)", "(b)", "(c)", "(d)", "(e)","(f)", "(g)", "(h)" ]
                options =''
                if dataset == 'ScienceQA' or dataset == 'ScienceQA_no_image':
                    
                    # question = "What is the answer about the above question?"
                    for i, choice in enumerate(choices):
                        option = option_num[i]+ " " + choice + " "
                        options +=option
                    task_definition  = task_definition + f"please choose the answer from \"{options}\" to the following question."
                    prompt = f"Please perform {task_name} task. {task_definition} {output_format}\n Question: {text}\nContext: {context}\n"
                    # question = 'What is the answer about the above question?'
                   
                prompt = prompt + f"Options: {options}\n" + "Answer:"
            
    elif setting == "few-shot":
        demo_string = ""
        for tup in demo_tuples:

            image_description = tup[-1]
            
            demo_string += f"\nImage Description: {image_description}\nSentence: {tup[0]}\nLabel:{tup[1]}\n"
       
        if task=="MABSA":   
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHere are demonstrations of this task.\n{demo_string}\nSentence: {text} Aspect: {aspect}\n"
            question = "what is the sentiment about the aspect based on the text-image pair?\n"
        elif task == "MSA":        
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHere are demonstrations of this task.\n{demo_string}\nSentence: {text}\n"
            question = "what is the sentiment about the text-image pair?\n"
        elif task == "MRE":
            head_entity = row['head_entity']
            head_cat = row['head_cat']
            tail_entity = row['tail_entity']
            tail_cat = row['tail_cat']
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\n{demo_string}\nSentence: {text} The head entity: {head_entity} belongs to {head_cat}; The tail entity: {tail_entity} belongs to {tail_cat}."
            question = "what has relation between the head entity and the tail entity based on the text-image pair?\n"
        elif task == "MHMR":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHere are demonstrations of this task.\n{demo_string}\nSentence: {text}\n"
            question = "whether or not the text-image pair contains the hate?\n"
        elif task == "MSR":
            prompt = f"Please perform {task_name} task.\n{task_definition}\n{output_format}\nHere are demonstrations of this task.\n{demo_string}\nSentence: {text}\n"
            question = "whether or not the text-image pair contains irony?\n"
            
        prompt = prompt + "Question: " + question + "Answer:" 
        
        
    else:
        raise NotImplementedError
    return prompt




def process_dataset(task, dataset, file_path, output_folder, model_name, setting, num_workers, train_path, shots, verbose=False, args=None):
    
    df = pd.read_csv(file_path)

    if setting in ["few-shot", "majority"]:
        train_df = pd.read_csv(train_path)
    else:
        train_df = None

    print(f"Predict on Task: {task}, Dataset: {dataset}")
    label_space = get_label_space(task, dataset)

    predictions = []
    predictions_original = []
    prompts = []
    prediction_indexes = []

    prompt_args = []
    if setting in ["zero-shot", "random", "majority"]:
        demo_tuples = None
    elif setting == "few-shot":
        demo_tuples = generate_fix_demo(train_df, task, dataset)
    else:
        raise NotImplementedError

    max_len = 0
    if task == 'QA':
        option_num = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    elif dataset == 'MVSA_Single' or dataset == 'MVSA_Multiple' or  dataset == 'Twitter_2015' or dataset == 'Twitter_2017':
        option_num = ["(a)", "(b)", "(c)"]
    elif dataset == 'MASAD' or dataset == 'MOSI_2' or dataset == 'MOSEI_2' or dataset == 'Sarcasm' or dataset == 'hate':
        option_num = ["(a)", "(b)"]
    elif dataset == "MOSEI_7" or dataset == "MOSI_7" or dataset == 'TumEmo':
        option_num = ['(a)', '(b)', '(c)', '(d)', "(e)", '(f)', '(g)', '(h)']
    elif dataset == 'MNRE':
        option_num = ['(a)', '(b)', '(c)', '(d)', "(e)", '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)']
    
    
    output_path = os.path.join(output_folder, f"prediction.csv")
    if setting in ["zero-shot", "few-shot"]:
        if model_name is not None:
            ##############################################load model#######################################################
            if model_name == 'chatgpt':
                if args.api_key is not None:
                    parallel_call = parallel_query_chatgpt_model
            elif model_name == 'text_flan-t5-xxl' or 'decapoda-llama'in args.model_name or 'meta-llama2' in args.model_name:
                tokenizer,  model, model_type= load_model(args)
                
            
            elif model_name == 'blip2_t5' or model_name == 'blip2_instruct_flant5xxl' or model_name == 'fromage' or model_name == 'openflamingo'  or model_name == 'mmgpt':
                model, model_type = load_model(args)
            elif model_name == 'LaVIN_7B' or model_name == 'LaVIN_13B':
                max_len = 0
                local_rank, world_size = setup_model_parallel()
                if local_rank > 0:
                    sys.stdout = open(os.devnull, "w")
                image_transforms=transforms.Compose(
                                                    [transforms.Resize((224, 224), 
                                                    interpolation=Image.BICUBIC),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
                
                model, model_type = load_model(args, local_rank, world_size)
            elif model_name == 'lynx_llm':
               
                model_type = 'finetune_lynx'
                config_path = f"multimodal_eval_main/models/lynx_llm/configs/{task}/{dataset}/LYNX_Prompt_type_{args.prompt_type}.yaml" 
                config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
                model = LynxBase(config=config, freeze_vit=config['freeze_vit'], freeze_llm=config['freeze_llm'], load_bridge=False)
                model = model.to(device)
                args.model_type = model_type
                
                for _, param in model.named_parameters():
                    param.requires_grad = False
                model.eval()
                print("### Creating datasets", flush=True)
                test_dataset = create_dataset('eval', config)

                start_time = time.time()
                print("### Start evaluating", flush=True)

                test_loader = create_loader([test_dataset],
                                            batch_size=[config['batch_size_test']],
                                            num_workers=[4],
                                            collate_fns=[test_dataset.collate_fn])[0]
            elif model_name == 'mplug_owl':
                tokenizer, model, image_processor, model_type = load_model(args)
            
            elif model_name == 'minigpt4':
                model, vis_processor, model_type = load_model(args)
                chat = minigpt4_Chat(model, vis_processor, device)
            
            elif model_name == 'llama_adapterv2':
                model, vis_processor, model_type = load_model(args)
            elif model_name == 'vpgtrans':
                model, vis_processor, model_type = load_model(args)
                chat = VPGTans_Chat(model, vis_processor, device)
            elif 'llava' in model_name:
                tokenizer, model, image_processor, model_type, mm_use_im_start_end = load_model(args)
                if mm_use_im_start_end:
                    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
                
            ##############################################load model#######################################################
                
            ##############################################generate prediction#######################################################   
            if model_name == 'lynx_llm':
                result = []
                with open(output_path, 'w') as fw:
                    tsv_w = csv.writer(fw, delimiter=',')
                    if task != 'QA':
                        tsv_w.writerow(['original_index', 'label_text', 'pred', 'prediction_original'])
                        for n, (idx, vision_input, input_ids, input_atts, label_texts) in tqdm(enumerate(test_loader)):
                            vision_input = vision_input.to(device, non_blocking=True)
                            input_ids = input_ids.to(device)
                            input_atts = input_atts.to(device)
                            # print('idx is {}'.format(idx))

                            text_outputs = model.generate(
                                vision_input=vision_input,
                                input_ids=input_ids, input_atts=input_atts,
                                use_nucleus_sampling=config.get('use_nucleus_sampling', False),
                                apply_lemmatizer=config['apply_lemmatizer'],
                                num_beams=config['num_beams'],
                                min_length=config['min_length'],
                                length_penalty=config.get('length_penalty', 1.0),
                                no_repeat_ngram_size=config.get('no_repeat_ngram_size', -1),
                                top_p=config.get('top_p', 0.9),
                                top_k=config.get('top_k', 3),
                                max_new_tokens=config.get('max_new_tokens', 10))

                            for i, label_text, pred_original in zip(idx, label_texts, text_outputs):
                                flag_set = set()
                                if 'MOSI_7' in dataset or dataset=='MOSEI_7':
                                    flag_set = set()
                                    if 'strongly negative' in pred_original:
                                        new_pred = 'strongly negative'
                                        flag_set.add(new_pred)
                                    if 'weakly negative' in pred_original:
                                        new_pred = 'weakly negative'
                                        flag_set.add(new_pred)
                                    if 'negative' in pred_original and 'strongly' not in pred_original and 'weakly' not in pred_original:
                                        new_pred = 'negative'
                                        flag_set.add(new_pred)
                                    if 'strongly positive' in pred_original:
                                        new_pred = 'strongly positive'
                                        flag_set.add(new_pred)
                                    if 'weakly positive' in pred_original:
                                        new_pred = 'weakly positive'
                                        flag_set.add(new_pred)
                                    if 'positive' in pred_original and 'strongly' not in pred_original and 'weakly' not in pred_original:
                                        new_pred = 'positive'
                                        flag_set.add(new_pred)
                                    if 'neutral' in pred_original:
                                        new_pred = 'neutral'
                                        flag_set.add(new_pred)
                                    if len(flag_set)==1: 
                                        new_pred= flag_set.pop()    
                                    else:
                                        new_pred= 'nan'
                                    predictions.append(new_pred)
                                else:
                                    flag_set = set()
                                    flag=0
                                    for label in label_space:
                                        if label in pred_original:
                                            flag_set.add(label)
                                    if len(flag_set)==1: 
                                        new_pred= flag_set.pop()    
                                    else:
                                        new_pred= 'nan'
                                tsv_w.writerow([i, label_text, new_pred, pred_original])
                                result.append({"index": i, "label": label_text, 'pred': new_pred, "pred_original": pred_original.strip()})
                    else:
                        tsv_w.writerow(['original_index', 'text', 'image' 'choices', 'answer_text', 'answer_index', 'prediction_index', 'pred', 'prediction_original'])
                        for n, (idx, vision_input, input_ids, input_atts, texts, images, choices, label_texts, label_indexes) in tqdm(enumerate(test_loader)):
                            vision_input = vision_input.to(device, non_blocking=True)
                            input_ids = input_ids.to(device)
                            input_atts = input_atts.to(device)
                            # print('idx is {}'.format(idx))

                            text_outputs = model.generate(
                                vision_input=vision_input,
                                input_ids=input_ids, input_atts=input_atts,
                                use_nucleus_sampling=config.get('use_nucleus_sampling', False),
                                apply_lemmatizer=config['apply_lemmatizer'],
                                num_beams=config['num_beams'],
                                min_length=config['min_length'],
                                length_penalty=config.get('length_penalty', 1.0),
                                no_repeat_ngram_size=config.get('no_repeat_ngram_size', -1),
                                top_p=config.get('top_p', 0.9),
                                top_k=config.get('top_k', 3),
                                max_new_tokens=config.get('max_new_tokens', 30))
                            for i, text, image, choices, label_text, label_index, pred_original in zip(idx, texts, images, choices, label_texts,  label_indexes, text_outputs):
                                flag_set = set()
                                index_set = set()
                                flag=0
                                for test_index, test_row in  df.iterrows():
                                    if str(test_row['original_index']) == str(i):
                                        choices = eval(test_row['choices'])
                                print(f"+++++++++++choices is {choices}, the length is {len(choices)}++++++++++++++++++++++")
                                for i_, choice in enumerate(choices):
                                    if (pred_original.lower() == choice.lower()):
                                        flag_set.add(choice)
                                        index_set.add(i_)
                                    if pred_original.lower() in choice.lower() and len(flag_set)==0:
                                        flag_set.add(choice)
                                        index_set.add(i_)
                                    if choice.lower() in pred_original.lower() and len(flag_set)==0:
                                        flag_set.add(choice)
                                        index_set.add(i_)
                                    if option_num[i_] in pred_original.lower() and len(flag_set)==0:
                                        flag_set.add(choice)
                                        index_set.add(i_) 
                                    
                                if len(flag_set)==1: 
                                    pred= flag_set.pop() 
                                    prediction_index = index_set.pop()  
                                else:
                                    pred= 'nan'
                                    prediction_index=-1
                                
                                
                                tsv_w.writerow([i, text, image, choice, label_text, label_index, prediction_index, pred, pred_original])
                                result.append({"index": i, 'text':text, 'image': image, 'choice': choice, 
                                               "label": label_text, 'label_index': label_index,
                                               'prediction_index': prediction_index, 'pred': pred,  
                                               "prediction_original": pred.strip()})
                
                print("### Prediction Results Save To: ", output_path, flush=True)
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('### Time {}'.format(total_time_str))
                prompt_sample = config['prompt']
            
            elif model_name == 'chatgpt':
                with open(output_path, 'a') as fww:
                    tsv_ww = csv.writer(fww, delimiter=',')
                    # tsv_w.writerow(['original_index', 'text', 'image', 'label_text', 'prediction'])
                    if task!='QA':
                        if dataset == "MOSI_7" or dataset == "MOSI_2":
                            tsv_ww.writerow(['original_index', 'text', 'image', 'label_scores', 'label_text', 'prediction', 'prediction_original'])
                        elif dataset == "MOSEI_2" or dataset == "MOSEI_7":
                            tsv_ww.writerow(['original_index', 'text', 'image', 'label_score', 'round_score', 'label_text', 'prediction', 'prediction_original'])
                        else:
                            tsv_ww.writerow(['original_index', 'text', 'image', 'label_text', 'prediction', 'prediction_original'])
                    else:
                        tsv_ww.writerow(['original_index', 'question', 'image', 'choices',  'hint',  'answer_text', 'answer'])
                    for index, row in tqdm(df.iterrows()):
                        print('index is {}'.format(row['original_index']))
                        prompt = generate_prompt(setting, task, dataset, label_space, row, demo_tuples, model_name, args.prompt_type, args)
                        # print('prompt is {}'.format((prompt)))
                        max_len = max(max_len, len(prompt.split()))
                        # if index == 0:
                        prompt_sample = prompt
                        pred_original = parallel_call(args.api_key, args.chatgpt_engine, prompt).lower()
                        if task !="QA":
                            
                            flag_set = set()
                            flag=0
                            for label in label_space:
                                if label in pred_original:
                                    flag_set.add(label)
                            if len(flag_set)==1: 
                                pred= flag_set.pop()    
                            else:
                                pred= 'nan'
                        
                            predictions.append(pred)
                        
                            if dataset == "MOSI_2" or dataset == "MOSI_7":
                                tsv_ww.writerow([row['original_index'], row['text'], row['image'], row['label_score'], row['label_text'],pred, pred_original])
                            elif dataset == "MOSEI_2" or dataset == "MOSEI_7":
                                tsv_ww.writerow([row['original_index'], row['text'], row['image'], row['label_score'], row['round_score'], row['label_text'],pred, pred_original])
                            else:
                                tsv_ww.writerow([row['original_index'], row['text'], row['image'], row['label_text'],pred, pred_original])
                        else:
                            pred=pred_original
                            flag_set = set()
                            index_set = set()
                            flag=0
                            choices = eval(row['choices'])
                            for i_, choice in enumerate(choices):
                                if (pred.lower() in choice.lower()) or (choice.lower() in (pred.lower()) or option_num[i_] in pred): #
                                    
                                    flag_set.add(pred)
                                    index_set.add(i_)
                            # print('++++++++++++++++++++++++++the flag_set is {}'.format(flag_set))
                            if len(flag_set)==1: 
                                pred= flag_set.pop() 
                                prediction_index = index_set.pop()  
                            else:
                                pred= 'nan'
                                prediction_index=-1
                            
                            
                            predictions.append(pred)
                            prediction_indexes.append(prediction_index)
                        
                            tsv_ww.writerow([row['original_index'], row['question'], row['image'], row['choices'],  row['hint'],  row['answer_text'], row['answer'], prediction_index, pred, pred_original])
                
                                
            else:   
                for index, row in tqdm(df.iterrows()):
                    # print('index is {}'.format(index))
                    prompt = generate_prompt(setting, task, dataset, label_space, row, demo_tuples, model_name, args.prompt_type, args)
                    if index==0:
                        print('prompt is {}'.format((prompt)))
                    ##read image
                    image_path = row['image']
                    
                    if model_name == 'text_flan-t5-xxl':
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
                        outputs = model.generate(input_ids)
                        pred = tokenizer.decode(outputs[0])
                        pred = pred.split("<pad>")[-1].strip().split('</s>')[0]
                        
                    elif 'decapoda-llama'in model_name or 'meta-llama2' in model_name:
                        batch = tokenizer(prompt, return_tensors="pt")
                        batch = {k: v.to(device) for k, v in batch.items()}
                        
                        with torch.no_grad():
                            # print(f'++++++++++++++++++the args.max_output_new_length is {args.max_output_new_length}++++++++++++')
                            outputs = model.generate(
                                                    **batch,
                                                    max_new_tokens=args.max_output_new_length,
                                                    do_sample=False,
                                                    top_p=args.top_p,
                                                    temperature=args.temperature,
                                                    min_length=1,
                                                    use_cache=False,
                                                    top_k=args.top_k,
                                                    repetition_penalty=args.repetition_penalty,
                                                    length_penalty=args.length_penalty,
                                                    )

                            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        
                    elif model_name == 'blip2_t5' or model_name == 'blip2_instruct_flant5xxl' or model_name == 'fromage' or model_name == 'openflamingo':
                        inputs = MultimodalSequence(
                        parts=[
                            MultimodalPart(content=image_path, is_image=True),
                            MultimodalPart(content=prompt),
                        ]
                        )
                        pred = model.run(inputs)
                    elif model_name == 'mmgpt':
                        pred = model(prompt=prompt, 
                            imgpaths=[image_path],
                            max_new_token=args.max_output_new_length, 
                            num_beams=1, 
                            temperature=1.0,
                            top_k=0, 
                            top_p=1.0, 
                            do_sample=False
                            )
                        
                    elif model_name == 'LaVIN_7B' or model_name == 'LaVIN_13B':
                        images = []
                        if image_path is not None:
                            image = Image.open(image_path).convert('RGB')
                            image = image_transforms(image)
                            indicator = 1
                        else:
                            image = torch.Tensor(torch.zeros(3, 224, 224).float())
                            indicator = 0
                        images.append(image.unsqueeze(0))
                        images=torch.cat(images,0)
                        pred =  model.generate(prompts=[prompt],
                                                    images=images,
                                                    indicators=[indicator],
                                                    max_gen_len=args.max_gen_len, temperature=args.generation_temperature, top_p=args.top_p,
                                                    n_feats=args.n_prompt)
                        pred=pred[0]
                    
                    elif model_name == 'mplug_owl':
                        raw_image = [Image.open(image_path)]
                        inputs = image_processor(text=[prompt], images=raw_image, return_tensors='pt')
                        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        generate_kwargs = {
                            'do_sample': True,
                            'top_k': 5,
                            'max_length': args.max_output_new_length
                        }
                        with torch.no_grad():
                            res = model.generate(**inputs, **generate_kwargs)
                        pred = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
                        
                    elif model_name == 'minigpt4':
                        

                        chat_state = minigpt4_CONV_VISION.copy()
                        ##read image
                        img_list = []
                        chat.upload_img(image_path, chat_state, img_list)
                        chat.ask(prompt, chat_state)

                        output_text = chat.answer(conv=chat_state,
                                                img_list=img_list,
                                                num_beams=1,
                                                temperature=0.01,
                                                max_new_tokens=args.max_output_new_length,
                                                max_length=2000)[0]
                        pred = output_text.replace("</s>", "")
                    
                    elif model_name == 'llama_adapterv2':
                        raw_image = cv2.imread(image_path)
                        raw_image = Image.fromarray(raw_image)
                        raw_image = vis_processor(raw_image).unsqueeze(0).to(device)
                        pred = model.generate(raw_image,[prompt])[0]
                    
                    elif model_name == 'vpgtrans':
                        raw_image = Image.open(image_path).convert('RGB')
                        # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

                        chat_state = VPGTans_CONV_VISION.copy()
                        img_list = []
                        chat_state.messages = []
                        llm_message = chat.upload_img(raw_image, chat_state, img_list)
                        chat.ask(prompt, chat_state)
                        pred = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=args.max_output_new_length, max_length=2000)[0].strip()
                    elif 'llava' in model_name:
                        raw_image = Image.open(image_path).convert('RGB')
                        vision_tower = model.get_model().vision_tower[0]
                        if vision_tower.device.type == 'meta':
                            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
                            model.get_model().vision_tower[0] = vision_tower
                        else:
                            vision_tower.to(device)
                        vision_config = vision_tower.config
                        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
                        vision_config.use_im_start_end = mm_use_im_start_end
                        if mm_use_im_start_end:
                            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
                        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
                        
                        if mm_use_im_start_end:
                            prompt = prompt + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
                        else:
                            prompt = prompt + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

                        if "v1" in args.llava_model_path.lower():
                            conv_mode = "llava_v1"
                        elif "mpt" in args.llava_model_path.lower():
                            conv_mode = "mpt_multimodal"
                        else:
                            conv_mode = "multimodal"

                        if args.conv_mode is not None and conv_mode != args.conv_mode:
                            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
                        else:
                            args.conv_mode = conv_mode
                        
                        conv = conv_templates[args.conv_mode].copy()
                        conv.append_message(conv.roles[0], prompt)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                        inputs = tokenizer([prompt])
                        image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
                        input_ids = torch.as_tensor(inputs.input_ids).to(device)

                        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                        keywords = [stop_str]
                        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                        
                        with torch.inference_mode():
                            output_ids = model.generate(
                                input_ids,
                                images=image_tensor.unsqueeze(0).half().to(device),
                                do_sample=True,
                                temperature=0.2,
                                max_new_tokens=512,
                                stopping_criteria=[stopping_criteria])
                       
                        input_token_len = input_ids.shape[1]
                        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                        if n_diff_input_output > 0:
                            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                        pred = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
               
                        if pred.endswith(stop_str):
                            pred = pred[:-len(stop_str)]
                ##############################################generate prediction#######################################################
                        
                        
                    predictions_original.append(pred)
                    
                    if model_name == 'LaVIN_7B' or model_name == 'LaVIN_13B' or model_name == 'mmgpt' or 'decapoda-llama'in model_name or 'meta-llama2' in model_name:
                        if args.prompt_type=='1':
                            str1= 'Label:'
                            index = pred.find(str1)
                            pred_original = pred[index:].lower()
                        elif args.prompt_type=="2" or args.prompt_type=="5" or args.prompt_type=="7":
                            if task != 'QA':
                                str1="Answer:"
                            else:
                                str1 = 'The answer is:'
                            index = pred.find(str1)
                            pred_original = pred[index:].lower()
                        elif args.prompt_type=="3" or args.prompt_type=="9" or args.prompt_type=="11":
                            str1="### Response:"
                            index = pred.find(str1)
                            pred_original = pred[index:].lower()
                        elif args.prompt_type=="4":
                            if task == 'MSC':
                                str1 = "what is the sentiment about the text-image pair?"
                            elif task=='MASC':
                                str1 = 'what is the sentiment about the aspect based on the text-image pair?'
                            elif task == "MHM":
                                str1 = "whether or not the text-image pair contains the hate?"
                            elif task == "Multimodal_Sarcasm_Detection":
                                str1 = "whether or not the text-image pair contains irony?"
                            elif task=="MNRE":
                                str1="what has relation between the head entity and the tail entity about the text-image pair?"
                            elif task == "QA":
                                str1 = "What is the answer about the above question?"
                            index=pred.find(str1)
                            pred_original = pred[index:].lower()
                        elif args.prompt_type=="6":
                            str1="AI:"
                            index = pred.find(str1)
                            pred_original = pred[index:].lower()
                        elif args.prompt_type=="8":
                            str1="Sentence:"
                            index = pred.find(str1)
                            pred_original = pred[index:].lower()
                        elif args.prompt_type=="10":
                            str1="<answer>"
                            index = pred.find(str1)
                            pred_original = pred[index:].lower()
                        if args.prompt_type =="3" or args.prompt_type =="5" or args.prompt_type =="9":
                            if dataset == "MVSA_Multiple" or dataset == "MVSA_Single" or dataset == "Twitter_2015" or dataset=="Twitter_2017":
                                if "(a)" in pred_original:
                                    pred_original = "neutral "
                                elif "(b)" in pred_original:
                                    pred_original = "negative "
                                elif "(c)" in pred_original:
                                    pred_original="positive "
                            elif dataset == "TumEmo":
                                ##Options: (a) angry (b) bored (c) calm (d) fear (e) happy (f) love (g) sad
                                if "(a)" in pred_original:
                                    pred_original = "angry "
                                elif "(b)" in pred_original:
                                    pred_original = "bored "
                                elif "(c)" in pred_original:
                                    pred_original="calm "
                                elif "(d)" in pred_original:
                                    pred_original = "fear "
                                elif "(e)" in pred_original:
                                    pred_original = "happy "
                                elif "(f)" in pred_original:
                                    pred_original = "love "
                                elif "(g)" in pred_original:
                                    pred_original = "sad "
                            elif dataset == "MASAD" or dataset == "MOSI_2"  or dataset == "MOSEI_2": 
                                # options = "(a) negative (b) positive"
                                if "(a)" in pred_original:
                                    pred_original = "negative "
                                elif "(b)" in pred_original:
                                    pred_original="positive "
                            elif dataset == "MSD" or dataset=="hate":
                                if "(a)" in pred_original:
                                    pred_original = "yes "
                                elif "(b)" in pred_original:
                                    pred_original="no "
                            elif dataset == "MOSI_7"  or dataset == "MOSEI_7":
                                if "(a)" in pred_original:
                                    pred_original = "negative "
                                elif "(b)" in pred_original:
                                    pred_original = "neutral "
                                elif "(c)" in pred_original:
                                    pred_original="positive "
                                elif "(d)" in pred_original:
                                    pred_original = "strongly negative"
                                elif "(e)" in pred_original:
                                    pred_original = "strongly positive "
                                elif "(f)" in pred_original:
                                    pred_original = "weakly negative "
                                elif "(g)" in pred_original:
                                    pred_original = "weakly positive"
                        pred = pred_original
                        
                    pred = pred.lower()
                    pred_list = pred.split(' ')
                    ### process your output, maybe you need to design the specific method to deal with your output of model.
                    if task =='QA':
                        flag_set = set()
                        index_set = set()
                        flag=0
                        choices = eval(row['choices'])
                        for i_, choice in enumerate(choices):
                            if pred.lower() in choice.lower():
                                flag_set.add(choice)
                                index_set.add(i_)
                        if len(flag_set)==0:
                            for i_, choice in enumerate(choices):
                                if choice.lower() in pred.lower():
                                    flag_set.add(choice)
                                    index_set.add(i_)
                        if len(flag_set)==0:
                            for i_, option_id in enumerate(option_num):
                                for pred_ in pred_list:
                                    if option_id in pred_:
                                        flag_set.add(choice)
                                        index_set.add(i_) 
                        if len(flag_set)==1: 
                            new_pred= flag_set.pop() 
                            prediction_index = index_set.pop()  
                        else:
                            new_pred= 'nan'
                            prediction_index=-1
                        
                        
                        predictions.append(new_pred)
                        prediction_indexes.append(prediction_index)
                    else:
                        flag_set = set()
                        if 'MOSI_7' in dataset or dataset=='MOSEI_7':
                            flag_set = set()
                            if 'strongly negative' in pred:
                                new_pred = 'strongly negative'
                                flag_set.add(new_pred)
                            if 'weakly negative' in pred:
                                new_pred = 'weakly negative'
                                flag_set.add(new_pred)
                            if 'negative' in pred and 'strongly' not in pred and 'weakly' not in pred:
                                new_pred = 'negative'
                                flag_set.add(new_pred)
                            if 'strongly positive' in pred:
                                new_pred = 'strongly positive'
                                flag_set.add(new_pred)
                            if 'weakly positive' in pred:
                                new_pred = 'weakly positive'
                                flag_set.add(new_pred)
                            if 'positive' in pred and 'strongly' not in pred and 'weakly' not in pred:
                                new_pred = 'positive'
                                flag_set.add(new_pred)
                            if 'neutral' in pred:
                                new_pred = 'neutral'
                                flag_set.add(new_pred)
                            if len(flag_set)==1: 
                                new_pred= flag_set.pop()    
                            else:
                                new_pred= 'nan'
                            predictions.append(new_pred)
                        else:
                            flag_set = set()
                            flag=0
                            for label in label_space:
                                if label in pred:
                                    flag_set.add(label)
                            if len(flag_set)==0:
                                for option in option_num:
                                    flag_set.add(label_space[option_num.index(option)])
                                    
                            if len(flag_set)==1: 
                                new_pred= flag_set.pop()    
                            else:
                                new_pred= 'nan'
                        
                            predictions.append(new_pred)
                    max_len = max(max_len, len(prompt.split()))
                    # if index == 0:
                    prompt_sample = prompt

                    prompt_args.append((model_type, prompt))
                    
                    
                    

            for args in prompt_args:
                prompts.append(args[1])
        else:
            for index, row in tqdm(df.iterrows()):
                prompt = generate_prompt(setting, task, dataset, label_space, row, demo_tuples)
                max_len = max(max_len, len(prompt.split()))
                if index == 0:
                    prompt_sample = prompt
                pred = generate_fake_data(task, dataset, label_space, row)
                prompts.append(prompt)
                predictions.append(pred)
    elif setting in ["random", "majority"]:
            if setting == "majority":
                most_common = train_df["label_text"].value_counts().idxmax()
            for index, row in tqdm(df.iterrows()):
                prompt_sample = ""
                if setting == "random":
                    pred = generate_fake_data(task, dataset, label_space, row)
                elif setting == "majority":
                    # should use train file
                    pred = most_common
                prompts.append("")
                predictions.append(pred)
    else:
        raise NotImplementedError

    # print(f"max_len: {max_len}")
    if verbose:
        print(prompt)
    
    if model_name != 'lynx_llm' and model_name !='chatgpt':
        if task == "QA":
            df['predictions_index'] = prediction_indexes
        df["prediction"] = predictions
        df['predictions_original'] = predictions_original
        df.to_csv(output_path, index=False)

    return prompt_sample
    


# Function to process the task and process datasets
def process_task(args, task, api_key, selected_datasets=None, ignored_datasets=None):

    setting = args.setting
    num_workers = args.num_workers
    shots = args.shots
    seed = args.seed
    model = args.model_name
    root_path = args.root_path
    test_path = args.test_path
    prompt_type = args.prompt_type

    task_folder = os.path.join(root_path, f"{task}")

    if setting in ["zero-shot", "random", "majority"]:
        output_task_folder = f"outputs/{setting}_{prompt_type}/model_{model}/seed_{seed}/{task}"
    elif setting == "few-shot":
        output_task_folder = f"outputs/{setting}/shot_{shots}/model_{model}/seed_{seed}/{task}"
    else:
        raise NotImplementedError

    prompt_samples = []
    dataset_names = []

    def check_entry(entry, selected_datasets, ignored_datasets):
        return entry.is_dir() and (selected_datasets is None or entry.name in selected_datasets) \
            and (ignored_datasets is None or entry.name not in ignored_datasets)

    entries = (entry for entry in sorted(os.scandir(task_folder), key=lambda e: e.name) if check_entry(entry, selected_datasets, ignored_datasets))
    for dataset in entries:
        if task=="QA":
            if not args.use_context:
                output_dataset_folder = os.path.join(output_task_folder, f'{dataset.name}_no_context')
            else:
                output_dataset_folder = os.path.join(output_task_folder, dataset.name)
        else:
            output_dataset_folder = os.path.join(output_task_folder, dataset.name)
        os.makedirs(output_dataset_folder, exist_ok=True)

        file_path = os.path.join(dataset.path, test_path)

        if setting in ["zero-shot", "random"]:
            train_path = None
        elif setting == "majority":
            train_path = os.path.join(f"csv/{task}/{dataset.name}", "train.csv")
        elif setting == "few-shot":
            train_path = os.path.join(dataset.path, f"shot_{shots}", f"seed_{seed}", "train.csv")
        else:
            raise NotImplementedError

        print(f'the output_dataset_folder is {output_dataset_folder}')
        if args.skip_runned:
            pred_file = os.path.join(output_dataset_folder, "prediction.csv")
            if os.path.exists(pred_file):
                print(f"{task} {dataset.name} skiped")
                continue

        prompt_sample = process_dataset(task, dataset.name, file_path, output_dataset_folder, model, setting, num_workers, train_path, shots, args=args)

        prompt_samples.append(prompt_sample)
        dataset_names.append(dataset.name)
    prompt_file = os.path.join(output_dataset_folder, "prompt.txt")
    with open(prompt_file, 'w') as f:
        for task_dataset, prompt in zip(dataset_names, prompt_samples):
            f.write('-'*100+'\n')
            f.write(f"{task}-{task_dataset}:\n{prompt}\n\n")


def main():
    # args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(args.selected_tasks)
    print(f"++++++++++++++++++The seed is {args.seed} and the prompt_type is {args.prompt_type}+++++++++++++++++++++++++++++++++++++++++++++")
   
    selected_tasks = eval(args.selected_tasks) if args.selected_tasks else ["sc", "mast", "absa"]
    selected_datasets = eval(args.selected_datasets) if args.selected_datasets else None
    ignored_datasets = eval(args.ignored_datasets) if args.ignored_datasets else None

    api_key = args.api

    for task in selected_tasks:
        if task == "QA":
            if args.use_context:
                print('We use context!!!!!!!!!!')
            else:
                print('We don\'t use context!!!!!!!!!!')
        process_task(args, task, api_key, selected_datasets, ignored_datasets)

if __name__ == "__main__":
    main()