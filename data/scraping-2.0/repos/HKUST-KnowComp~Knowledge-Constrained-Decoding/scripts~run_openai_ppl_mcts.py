import json
import os
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, DataCollatorWithPadding
from peft import PeftModel, LoraConfig, get_peft_model

from kcd.text_data import load_text_data
from kcd.attribute_classifier.attribute_classifier_model import DoubleHeadModel
from kcd.openai_module import OpenAIModel, OpenAIAPIParameters, MockOpenAIModel
from kcd.classifier_guidance.ppl_mcts import PplMCTSConfig
from kcd.classifier_guidance.openai_ppl_mcts import OpenAIMCTS
from kcd.configs import GenerationConfig


@dataclass
class ExperimentArgs:
    lm_name: str = field(default="gpt2-xl")
    openai_model_name: str = 'text-davinci-003'
    num_labels: int = field(default=2)
    attr_idx: int = 1
    dataset: str = 'wow'
    use_kilt_format: bool = False
    test_data_path: str = field(default="data/pplm_prompts.csv")
    output_path: str = 'generations/ppl_mcts'
    disc_name: str = ''
    instruction_model: str = 'basic'  # choices=['basic', 'openai', 'alpaca']
    load_peft_checkpoint: str = None
    load_checkpoint: str = None
    load_classifier: str = field(default=None)
    batch_size: int = 1
    continue_from: int = 0
    human_indices: str = None

    max_num_gen: int = -1

    mock_debug: bool = False


def main():
    parser = HfArgumentParser([ExperimentArgs, PplMCTSConfig, GenerationConfig])
    args, mcts_args, gen_cfg = parser.parse_args_into_dataclasses()

    print('cuda available:', torch.cuda.is_available())
    ################### load discriminator ####################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_kwargs = {
        'device_map': 'auto',
        'torch_dtype': torch.float16,
    }
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left"  # only for causal LM
    tokenizer.pad_token = tokenizer.eos_token

    model = DoubleHeadModel.from_pretrained(args.lm_name,
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

    ############################## OPENAI MODEL ###############################
    if args.mock_debug:
        openai_model = MockOpenAIModel()
    else:
        openai_model = OpenAIModel(args.openai_model_name)
    parameters = OpenAIAPIParameters(
        max_tokens=1,  # no lookahead
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        logprobs=5)
    ############################## Data ########################################
    print("loading dataset")
    dataset = load_text_data(path=args.test_data_path,
                             instruction_model=args.instruction_model,
                             task='completion',
                             use_kilt_format=args.use_kilt_format,
                             tokenize=True,
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
    print("dataset loaded")
    ############################### MCTS #######################################
    batch_size = args.batch_size
    MCTS = OpenAIMCTS(mcts_args,
                      tokenizer,
                      openai_model,
                      parameters,
                      model,
                      batch_size=batch_size,
                      top_k=6,
                      num_labels=args.num_labels,
                      unused_token_id=tokenizer.unk_token_id,
                      device=device)

    samples_pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Samples generated")

    ###################### out file configuration ##############################
    os.makedirs(args.output_path, exist_ok=True)
    dataset_name = args.dataset
    if args.use_kilt_format:
        dataset_name = f'{dataset_name}-kilt'
    out_fname = os.path.join(
        args.output_path, f'{dataset_name}-{args.lm_name.replace("/", "-")}-{args.disc_name}.jsonl')
    fout = open(out_fname, 'a')
    ####################### Generation Loop ####################################
    for i, batch in samples_pbar:
        batch = batch.to(device)
        labels = torch.zeros(batch['input_ids'].shape[0], args.num_labels, device=device)
        labels[:, args.attr_idx] = 1

        MCTS.set_labels(labels)
        _, decoded_ids = MCTS.search(batch, tokens_to_generate=gen_cfg.max_new_tokens)

        for j, (prompt_ids, response_ids) in enumerate(zip(batch['input_ids'], decoded_ids)):
            seqlen = len(prompt_ids)
            response_ids = response_ids[seqlen:]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            result = {
                'prompt': prompt,
                'response': response,
                'index': i * batch_size + j + args.continue_from,
                'token_usage': MCTS.lm_step.token_usages[j],
            }
            if indices is not None:
                result['index'] = indices[result['index']]
            result['gold'] = text_dataset['completion'][i * batch_size + j + args.continue_from]
            fout.write(json.dumps(result) + '\n')
    fout.close()


if __name__ == "__main__":
    main()
