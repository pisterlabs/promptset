import json
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import HfArgumentParser, AutoTokenizer, DataCollatorWithPadding
from peft import PeftModel, LoraConfig, get_peft_model

from kcd.attribute_classifier.attribute_classifier_model import DoubleHeadModel
from kcd.token_classifier.model import T5DoubleHeadModel
from kcd.text_data import load_text_data
from kcd.classifier_guidance import load_generate_fn
from kcd.openai_module import OpenAIModel, OpenAIAPIParameters, MockOpenAIModel
from kcd.configs import GenerationConfig


@dataclass
class ExperimentArgs:
    test_data_path: str = 'data/wow-dev-kilt-processed.jsonl'
    output_path: str = 'generations/fudge'
    model_name: str = "google/flan-t5-xl"
    openai_model_name: str = 'text-davinci-003'
    dataset: str = 'wow'
    use_kilt_format: bool = False
    instruction_model: str = 'basic'  # choices=['basic', 'openai', 'alpaca']
    guidance_method: str = 'openai_fudge'  # choices=['fudge', 'nado']
    disc_name: str = ''
    use_logit_bias: bool = False
    pre_post_guidance: bool = False
    propose_topk: int = 50

    num_labels: int = 2
    load_checkpoint: str = None
    load_peft_checkpoint: str = None
    load_classifier: str = None
    human_indices: str = None

    batch_size: int = 1
    continue_from: int = 0
    max_num_gen: int = -1

    mock_debug: bool = False


def main():
    parser = HfArgumentParser((ExperimentArgs, GenerationConfig))
    args, gen_cfg = parser.parse_args_into_dataclasses()

    print('cuda available:', torch.cuda.is_available())
    # TODO: token space / wordpiece issue with understanding chatGPT output.
    if args.openai_model_name == 'gpt-3.5-turbo':
        raise NotImplementedError("chatGPT not implemented yet")
    ################### load discriminator ####################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_kwargs = {
        'device_map': 'auto',
        'torch_dtype': torch.float16,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"  # only for causal LM
    tokenizer.pad_token = tokenizer.eos_token

    if 't5' in args.model_name:
        model = T5DoubleHeadModel.from_pretrained(args.model_name,
                                                  output_hidden_states=True,
                                                  use_cache=True,
                                                  num_labels=args.num_labels,
                                                  pool_method='last',
                                                  load_in_8bit=True,
                                                  **load_kwargs)
    else:
        model = DoubleHeadModel.from_pretrained(args.model_name,
                                                output_hidden_states=True,
                                                use_cache=True,
                                                num_labels=args.num_labels,
                                                pool_method='last',
                                                **load_kwargs)

    if args.load_peft_checkpoint:
        model = PeftModel.from_pretrained(model, args.load_peft_checkpoint)
    if args.load_checkpoint:
        peft_config_path = os.path.join(os.path.dirname(args.load_checkpoint), 'adapter_model')
        peft_config = LoraConfig.from_pretrained(peft_config_path)
        model = get_peft_model(model, peft_config)
        incompatible = model.load_state_dict(torch.load(args.load_checkpoint), strict=False)
        assert (len(incompatible.missing_keys) == 1 and
                incompatible.missing_keys[0].endswith('lm_head.weight'))
    if args.load_classifier:
        ckpt = torch.load(args.load_classifier)
        ckpt = {k.replace('score.', ''): v for k, v in ckpt.items() if 'score' in k}
        model.score.load_state_dict(ckpt, strict=True)

    task = 'chat' if args.openai_model_name == 'gpt-3.5-turbo' else 'completion'
    if args.mock_debug:
        openai_model = MockOpenAIModel()
    else:
        openai_model = OpenAIModel(args.openai_model_name, task=task)

    # load guidance criteria
    if args.guidance_method == 'openai_fudge':
        parameters = OpenAIAPIParameters(
            max_tokens=1,  # no lookahead
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            logprobs=5)
        generate_fn = load_generate_fn(args.guidance_method,
                                       openai_model=openai_model,
                                       model=model,
                                       tokenizer=tokenizer,
                                       max_new_tokens=gen_cfg.max_new_tokens,
                                       k=6,
                                       pre_post_guidance=args.pre_post_guidance,
                                       use_logit_bias=args.use_logit_bias,
                                       propose_topk=args.propose_topk,
                                       parameters=parameters)
    else:
        raise NotImplementedError(f"Guidance method {args.guidance_method} not implemented")

    ############################## Data ########################################
    print("loading dataset")
    dataset = load_text_data(path=args.test_data_path,
                             instruction_model=args.instruction_model,
                             task='completion',
                             use_kilt_format=args.use_kilt_format,
                             tokenize=True,
                             add_trailing_newline='t5' not in args.model_name,
                             tokenizer=tokenizer,
                             no_label=True)
    text_dataset = load_text_data(path=args.test_data_path,
                                  instruction_model=args.instruction_model,
                                  task='completion',
                                  use_kilt_format=args.use_kilt_format,
                                  tokenize=False)
    indices = None
    if args.max_num_gen > 0:
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), args.max_num_gen)
        dataset = dataset.select(indices)
        text_dataset = text_dataset.select(indices)

    if args.human_indices:
        with open(args.human_indices) as f:
            indices = [int(i) for i in f.readlines()]
        dataset = dataset.select(indices)
        text_dataset = text_dataset.select(indices)
    if args.continue_from > 0:
        dataset = dataset.select(range(args.continue_from, len(dataset)))
        text_dataset = text_dataset.select(range(args.continue_from, len(text_dataset)))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    ###################### out file configuration ##############################
    os.makedirs(args.output_path, exist_ok=True)
    dataset_name = args.dataset
    if args.use_kilt_format:
        dataset_name = f'{dataset_name}-kilt'
    out_fname = os.path.join(
        args.output_path,
        f'{dataset_name}-{args.model_name.replace("/", "-")}-{args.disc_name}.jsonl')
    fout = open(out_fname, 'a')

    ####################### Generation Loop ####################################
    samples_pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Samples generated")
    for i, batch in samples_pbar:
        batch = batch.to(device)

        decoded_ids, usage, success = generate_fn(batch)
        if not success:
            print("the generation failed abruptly.")

        for j, (prompt_ids, response_ids) in enumerate(zip(batch['input_ids'], decoded_ids)):
            seqlen = len(prompt_ids)
            response_ids = response_ids[seqlen:]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            result = {
                'prompt': prompt,
                'response': response,
                'index': i * args.batch_size + j + args.continue_from,
                'token_usage': usage[j].item(),
                'success': success,
            }
            if indices is not None:
                result['index'] = indices[result['index']]
            result['gold'] = text_dataset['completion'][i * args.batch_size + j +
                                                        args.continue_from]
            fout.write(json.dumps(result) + '\n')
    fout.close()


if __name__ == "__main__":
    main()
