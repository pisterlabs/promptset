def get_model(model_name, key_file=None, checkpoint_path=None):
    if checkpoint_path!=None:
        # Loading fine-tuned OPT models for evaluation
        encoder = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
        model = OPTForCausalLM.from_pretrained("facebook/opt-125m").cuda().eval()
        ckpt = torch.load(checkpoint_path)['state_dict']
        ckpt_new={}
        for key, value in ckpt.items():
            if 'model' in key:
                ckpt_new[key[6:]] = value
        model.load_state_dict(ckpt_new, strict=False)
        name = checkpoint_path
    elif model_name.lower() in ['gpt2', 'gpt2-s', 'gpt2-small', 'gs', 's', 'small']:
        # GPT-2 Small
        model   = GPT2LMHeadModel.from_pretrained('gpt2').cuda().eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2')
        name    = 'G-S'
    elif model_name.lower() in ['gpt2-m', 'gpt2-medium', 'gm', 'm', 'medium']:
        # GPT-2 Medium
        model   = GPT2LMHeadModel.from_pretrained('gpt2-medium').cuda().eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2-medium')
        name    = 'G-M'
    elif model_name.lower() in ['gpt2-l', 'gpt2-large', 'gl', 'l', 'large']:
        # GPT-2 Large
        model   = GPT2LMHeadModel.from_pretrained('gpt2-large').cuda().eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2-large')
        name    = 'G-L'
    elif model_name.lower() in ['gpt2-xl', 'gxl', 'xl', 'extra-large']:
        # GPT-2 XL
        model   = GPT2LMHeadModel.from_pretrained('gpt2-xl').cuda().eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2-xl')
        name    = 'G-XL'
    elif model_name.lower() in ['gptj']:
        # GPT-J 6B
        # model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda().eval()
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        model.parallelize()
        model.eval()
        # model   = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda().eval()
        encoder = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        name    = 'G-J'
    elif model_name.lower() == 'opt-125m':
        encoder = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
        model = OPTForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto")
        name = 'OPT-125m'
    elif model_name.lower() == 'opt-350m':
        encoder = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
        model = OPTForCausalLM.from_pretrained("facebook/opt-350m", device_map="auto")
        name = 'OPT-350m'
    elif model_name.lower() == 'opt-1.3b':
        encoder = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")
        model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b", device_map="auto")
        name = 'OPT-1.3b'
    elif model_name.lower() == 'opt-2.7b':
        encoder = GPT2Tokenizer.from_pretrained("facebook/opt-2.7b")
        model = OPTForCausalLM.from_pretrained("facebook/opt-2.7b", device_map="auto")
        name = 'OPT-2.7b'
    elif model_name.lower() == 'opt-6.7b':
        encoder = GPT2Tokenizer.from_pretrained("facebook/opt-6.7b")
        model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", device_map="auto")
        name = 'OPT-6.7b'
    elif model_name.lower() == 'opt-13b':
        encoder = GPT2Tokenizer.from_pretrained("facebook/opt-13b")
        model = OPTForCausalLM.from_pretrained("facebook/opt-13b", device_map="auto")
        name = 'OPT-13b'
    elif model_name.lower() == 'opt-30b':
        encoder = GPT2Tokenizer.from_pretrained("facebook/opt-30b")
        model = OPTForCausalLM.from_pretrained("facebook/opt-30b", device_map="auto")
        name = 'OPT-30b'
    elif model_name.lower() == 'opt-66b':
        encoder = GPT2Tokenizer.from_pretrained("facebook/opt-66b")
        model = OPTForCausalLM.from_pretrained("facebook/opt-66b", device_map="auto")
        name = 'OPT-66b'
    elif model_name in ['T0_3B', 'T0']:
        model_name ="bigscience/"+model_name
        # T0
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
        if args.tokenizer_name:
            encoder = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        else:
            encoder = AutoTokenizer.from_pretrained(model_name, use_fast=not args.use_slow_tokenizer)
        if encoder.pad_token is None:
            for token in [encoder.eos_token, encoder.bos_token, encoder.sep_token]:
                if token is not None:
                    encoder.pad_token = token
            if encoder.pad_token is None:
                raise ValueError("Please define a pad token id.")
        # model = ModelBase.from_config(
        #     config=config,
        #     model_name_or_path=model_name,
        #     checkpoint_path=args.checkpoint_path,
        #     parallelize=args.parallelize
        # )
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.parallelize()
        model.eval()
        if model_name=='T0_3B':
            name = 'T0-3B'
        else:
            name = 'T0-11B'
    elif model_name.lower() == 'ada' or \
         model_name.lower() == 'babbage' or \
         model_name.lower() == 'curie' or \
         model_name.lower() == 'davinci' or \
         model_name.lower() == 'text-ada-001' or \
         model_name.lower() == 'text-babbage-001' or \
         model_name.lower() == 'text-curie-001' or \
         model_name.lower() == 'text-davinci-002':
        # GPT-3
        model = name = model_name
        encoder = None
        import openai
        # with open(key_file) as f:
        #     api_key = f.read().strip()
        openai.api_key = "Insert your openai key here"
    else:
        raise ValueError(f'No model {model_name}')
    return model, encoder, name

def get_examples(dataset_name, split, stem, n_shot, variant, dataset_config, prompt_name, model_name, icl=0):
    if args.promptsource:
        from data_loaders import load_examples_promptsource
        examples = load_examples_promptsource(args.use_csv, f'{stem}copa-{split}.xml', dataset_name, dataset_config, prompt_name, model_name, icl)
        closed_label_space = True
    else:
        if dataset_name == 'copa':
            from data_loaders import load_examples_copa
            examples = load_examples_copa(f'{stem}copa-{split}.xml')
            closed_label_space = False
        elif dataset_name == 'copa-rev':
            from data_loaders import load_examples_copa_rev
            examples = load_examples_copa_rev(f'{stem}copa-{split}.xml')
            closed_label_space = False
        elif dataset_name == 'storycloze':
            from data_loaders import load_examples_storycloze
            examples = load_examples_storycloze(f'{stem}{split}.tsv')
            closed_label_space = False
        elif dataset_name == 'hellaswag':
            from data_loaders import load_examples_hellaswag
            examples = load_examples_hellaswag(f'{stem}dev.jsonl')
            closed_label_space = False
        elif dataset_name == 'race-m' or \
            dataset_name == 'race-h':
            from data_loaders import load_examples_race
            version = 'high' if dataset_name == 'race-h' else 'middle'
            examples = load_examples_race(stem, split, version)
            closed_label_space = False
        elif dataset_name == 'arc-easy' or \
            dataset_name == 'arc-challenge':
            from data_loaders import load_examples_arc
            examples = load_examples_arc(f'{stem}{split}.jsonl')
            closed_label_space = False
        elif dataset_name == 'obqa':
            from data_loaders import load_examples_obqa
            examples = load_examples_obqa(f'{stem}{split}.jsonl')
            closed_label_space = False
        elif dataset_name == 'cqa':
            from data_loaders import load_examples_cqa
            if args.split == 'test':
                raise NotImplementedError("CSQA does not release test answers directly, please do not spam their leaderboard either :)")
            else:
                examples = load_examples_cqa(f'{stem}{split}.jsonl')
            closed_label_space = False
        elif dataset_name == 'boolq':
            from data_loaders import load_examples_boolq
            examples = load_examples_boolq(f'{stem}dev.jsonl')
            closed_label_space = True
        elif dataset_name == 'rte':
            from data_loaders import load_examples_rte
            examples = load_examples_rte(f'{stem}dev.jsonl')
            closed_label_space = True
        elif dataset_name == 'cb':
            from data_loaders import load_examples_cb
            examples = load_examples_cb(f'{stem}dev.jsonl')
            closed_label_space = True
        elif dataset_name == 'sst-2':
            from data_loaders import load_examples_sst2, load_examples_sst2_variants
            if n_shot > 0:
                examples = load_examples_sst2(f'{stem}{split}.tsv', f'{stem}/train.tsv', n_shot)
            elif variant is not None:
                examples = load_examples_sst2_variants(f'{stem}{split}.tsv', variant)
            else:
                examples = load_examples_sst2(f'{stem}{split}.tsv')
            closed_label_space = True
        elif dataset_name == 'sst-5':
            from data_loaders import load_examples_sst5
            examples = load_examples_sst5(f'{stem}{split}.tsv')
            closed_label_space = True
        elif dataset_name == 'agn':
            from data_loaders import load_examples_agn
            examples = load_examples_agn(f'{stem}{split}.csv')
            closed_label_space = True
        elif dataset_name == 'trec':
            split = 'train' if split == 'dev' else split
            from data_loaders import load_examples_trec
            examples = load_examples_trec(f'{stem}{split}.txt')
            closed_label_space = True
        else:
            raise Exception(f'{dataset_name} not yet supported..')


    return examples, closed_label_space


if __name__ == '__main__':
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTJForCausalLM, OPTForCausalLM, T5ForConditionalGeneration
    import transformers
    from accelerate import Accelerator
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        default_data_collator,
    )
    from utils import score, score_T0
    import argparse
    import random
    import numpy as np
    import torch
    import os
    import pdb
    
    import logging
    import json
    import pandas as pd
    import datasets
    from datasets import load_dataset, load_metric
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from promptsource.promptsource.templates import DatasetTemplates
    from t0.data_collator import DataCollatorForMultipleChoice
    from t0.model import ModelBase
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='xl')
    parser.add_argument('--n-shot', type=int, default=0)
    parser.add_argument('--variant', type=int, default=None)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key', type=str, default='api.key')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--prompt_name', type=str, default=None)
    parser.add_argument('--prompt_list', type=list, default=None)
    parser.add_argument('--use_csv', action='store_true')

    #T0 settings: need extra essential parameter, template_name
    parser.add_argument('--promptsource', action='store_true')
    parser.add_argument('--max_input_length',type=int,default=512)
    parser.add_argument('--max_output_length',type=int,default=32)
    parser.add_argument('--eos_token', type=bool, default=False)
    parser.add_argument('--n_prefix',type=int,default=100)
    parser.add_argument('--icl',type=int, default=0)
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="If passed, will load checkpoint",
    )
    args = parser.parse_args()
    if args.promptsource and args.prompt_name==None:
        raise ValueError("Please define a prompt name when using promptsource.")
    args = parser.parse_args()
    print(args)

    if args.debug:
        pdb.set_trace()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.checkpoint_path:
        ckpt_list = []
        ckpt_list_ = os.listdir(args.checkpoint_path)
        for c in ckpt_list_:
            ckpt_list.append(args.checkpoint_path + '/' + c)
    else:
        ckpt_list = None

    if ckpt_list == None:
        model, encoder, name = get_model(args.model, args.key)
        if args.dataset.endswith('-rev'):
            stem = f'data/{args.dataset[:-4]}'
        else:
            stem = f'data/{args.dataset}'
        if args.dataset_config!=None:
            stem+=("/"+args.dataset_config)
        if args.prompt_name!=None:
            stem+=("/"+args.prompt_name)
        stem +="/"
        examples, closed_label_space = get_examples(args.dataset, args.split, stem, args.n_shot, args.variant, args.dataset_config, args.prompt_name, args.model, args.icl)
        # if args.sample:
        #     assert(args.sample <= len(examples))
        #     examples = random.sample(examples, args.sample)
        if "T0" in args.model:
            accs, accs_50 = score_T0(model, name, encoder, examples, stem, args.split, args.batch, args)
        else:
            accs, accs_50 = score(model, name, encoder, examples, stem, args.split, args.batch, args.prompt_name, args.icl)
        print(f'{name} gets {accs}% on {args.dataset}')
        print(f'{name} gets {accs_50}% on {args.dataset} on 50 set.')
    else:
        for ckpt in ckpt_list:
            model, encoder, name = get_model(ckpt, args.model, args.key)
            if args.dataset.endswith('-rev'):
                stem = f'data/{args.dataset[:-4]}'
            else:
                stem = f'data/{args.dataset}'
            if args.dataset_config!=None:
                stem+=("/"+args.dataset_config)
            if args.prompt_name!=None:
                stem+=("/"+args.prompt_name)
            stem +="/"
            examples, closed_label_space = get_examples(args.dataset, args.split, stem, args.n_shot, args.variant, args.dataset_config, args.prompt_name, args.model, args.icl)
            # if args.sample:
            #     assert(args.sample <= len(examples))
            #     examples = random.sample(examples, args.sample)
            if "T0" in args.model:
                accs, accs_50 = score_T0(model, name, encoder, examples, stem, args.split, args.batch, args)
            else:
                accs, accs_50 = score(model, name, encoder, examples, stem, args.split, args.batch, args.prompt_name, args.icl)
            print(f'{name} gets {accs}% on {args.dataset}')
            print(f'{name} gets {accs_50}% on {args.dataset} on 50 set.')