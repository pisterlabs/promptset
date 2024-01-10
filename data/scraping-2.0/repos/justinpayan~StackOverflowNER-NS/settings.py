import os
import json
import argparse
import logging
import datetime
logger = logging.getLogger(__name__)

import GPUtil
# from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config, CONFIG_NAME
import torch

import pathlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILL_VAL = -1
LEN_FACTOR = 1.163
MEMORY_FACTOR = {
    "finetune": 0.58,
    "multitask": 0.58,
    "lll": 0.35,
    "ewc": 0.30,
    "mas": 0.18,
    "gem": 0.50,
}
TURING_ARCHS = {'Tesla V100', '2080 Ti'}
MODEL_CLASSES = {
    'gpt2': (GPT2Model, GPT2Tokenizer, GPT2Config),
    # 'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig),
}
SAVE_NAME = 'model-'
FINAL_SAVE_NAME = 'model-finish'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam_epsilon", default=1e-4, type=float)
    parser.add_argument("--add_task_tokens", action="store_true")
    parser.add_argument("--add_ent_tokens", action="store_true")
    parser.add_argument("--ic", action="store_true")
    parser.add_argument("--use_task_in_ner", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--decay_style", type=str, default="linear")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--real_sample", action="store_true")
    parser.add_argument("--unbound", type=int, default=0)
    parser.add_argument("--gen_lm_sample_percentage", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--lm_lambda", type=float, default=0.25)
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=9)
    parser.add_argument("--min_batch_size", type=int, default=4)
    parser.add_argument("--min_n_steps", type=int, default=1500)
    parser.add_argument("--model_dir_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt2", choices=["gpt2", "openai-gpt"])
    parser.add_argument("--model_base_dir", type=str, default="")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_train_epochs", type=int, default=3)
    parser.add_argument("--dynamic_epochs", action="store_true")
    parser.add_argument("--n_warmup_ratio", type=float, default=0.005)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--use_sep", action="store_true")
    parser.add_argument("--reg_lambda", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_train_type", type=str, default="lll", choices=["lll","finetune","multitask","mas","ewc","gem"])
    parser.add_argument("--tasks", nargs='+', default=["conll2003"])
    parser.add_argument("--skip_tasks", nargs='+')
    parser.add_argument("--temperature_lm", type=float, default=1.0)
    parser.add_argument("--temperature_ner", type=float, default=1.0)
    parser.add_argument("--test_batch_size", type=int, default=0)
    parser.add_argument("--tokens_weight", type=float, default=5)
    parser.add_argument("--top_k_lm", type=int, default=20)
    parser.add_argument("--top_k_ner", type=int, default=20)
    parser.add_argument("--top_p_lm", type=float, default=0.)
    parser.add_argument("--top_p_ner", type=float, default=0.)
    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--qp_margin", type=float, default=0.5)
    parser.add_argument("--label_map_file", type=str, default="")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--test_last_only", action="store_true")
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--generated_csv", type=str, default="")
    parser.add_argument("--gpt_loc", type=str, default="")
    parser.add_argument("--original_dataset", type=str, default="")
    parser.add_argument("--browsing", type=int, default=0)
    parser.add_argument("--short_exs_debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.logging_steps = 1
        torch.manual_seed(0)
        torch.backends.cudnn.deterministric = True

    args.model_dir_root = os.path.join(args.model_dir_root, args.model_name,
            args.seq_train_type, "{}_{}_{}_{}".format("_".join(args.tasks),
                args.gen_lm_sample_percentage, args.lm_lambda, args.learning_rate) if "lll" in args.seq_train_type else "_".join(args.tasks))

    if args.real_sample:
        args.model_dir_root += "_real"
    if args.add_task_tokens and not args.use_task_in_ner:
        args.model_dir_root += "_tasktokens"
    if args.use_task_in_ner:
        args.model_dir_root += "_tasksner"
    if args.add_ent_tokens:
        args.model_dir_root += "_enttokens"
    if args.ic:
        args.model_dir_root += "_ic"
        if args.seq_train_type == "lll":
            args.model_dir_root += "_improvedgen_with_ic_control"
    elif args.seq_train_type == "lll":
        args.model_dir_root += "_improvedgen"

    # args.device_ids = GPUtil.getAvailable(maxLoad=0.05, maxMemory=0.05, limit=args.n_gpus)
    args.device_ids = [0]
    # print(args.device_ids)
    if not args.browsing and len(args.device_ids) == 0:
        logger.error('No available GPUs!')
        raise NotImplementedError("No CPU mode available!")

    if len(args.device_ids) < args.n_gpus:
        logger.warning('Available number of GPU = {} < n_gpus = {}'.format(len(args.device_ids), args.n_gpus))
        args.n_gpus = len(args.device_ids)
        logger.warning('Continue training with {} GPUs'.format(args.n_gpus))

    if not args.browsing:
        print(args.device_num)
        print(GPUtil.getAvailable(maxLoad=0.05, maxMemory=0.05, limit=args.n_gpus))
        # torch.cuda.set_device(args.device_ids[args.device_num])

        gpus = GPUtil.getGPUs()
        # gpu_names = [gpus[device_id].name for device_id in args.device_ids]
        gpu_names = [gpus[args.device_num].name]
        if not all(any(turing_arch in gpu_name for turing_arch in TURING_ARCHS) for gpu_name in gpu_names):
            logger.warning('Not all gpus support fp16 training! Will use fp32 instead.')
            args.fp32 = True
        if args.model_name == "openai-gpt":
            args.fp32 = True  # openai-gpt currently doesn't support fp16
        if not args.fp32:
            global MEMORY_FACTOR
            MEMORY_FACTOR = dict([k, v*1.4] for k, v in MEMORY_FACTOR.items())
        # args.memory_sizes = [gpus[device_id].memoryTotal for device_id in args.device_ids]
        args.memory_sizes = [gpus[args.device_num].memoryTotal]
        # args.memory_sizes[0] = args.memory_sizes[0] * (1 - 0.04 * (args.n_gpus-1))
        # for i in range(1, args.n_gpus):
        #     args.memory_sizes[i] = args.memory_sizes[i] * 1.04
        if args.train_batch_size <= 0:
            args.train_batch_size = [int(memory_size * MEMORY_FACTOR[args.seq_train_type]) for memory_size in args.memory_sizes]
        if args.test_batch_size <= 0:
            args.test_batch_size = [int(memory_size * MEMORY_FACTOR[args.seq_train_type]) for memory_size in args.memory_sizes]

    special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
    if args.use_sep:
        special_tokens["sep_token"] = '__sep__'
    if args.ic:
        special_tokens["ic"] = '__intent__'

    model_class, tokenizer_class, config_class = MODEL_CLASSES[args.model_name]
    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    tokenizer.add_tokens(list(special_tokens.values()))
    special_token_ids = {k:tokenizer.convert_tokens_to_ids(v) for k,v in special_tokens.items()}

    model_config = config_class.from_pretrained(args.model_name)
    model_config.vocab_size = len(tokenizer)
    tokens_weight = torch.ones([model_config.vocab_size], dtype=torch.float).cuda()
    tokens_weight[special_token_ids["ans_token"]] = args.tokens_weight
    if args.use_sep:
        tokens_weight[special_token_ids["sep_token"]] = args.tokens_weight
    if args.ic:
        tokens_weight[special_token_ids["ic"]] = args.tokens_weight

    args.max_len = model_config.n_positions

    data_attrs_path = os.path.join(BASE_DIR, "data_attrs.json")
    assert os.path.exists(data_attrs_path)
    with open(data_attrs_path, "r") as f:
        data_attrs = json.load(f)

    if args.seq_train_type == "multitask":
        args.n_train_epochs = {'_'.join(args.tasks): args.n_train_epochs}
    elif args.unbound:
        pass
    else:
        if "gem" in args.seq_train_type:
            args.memory_data = []
        if args.dynamic_epochs:
            data_sizes = {task: data_attrs[task]["train"]["data_size"] for task in args.tasks}
            max_total_data_size = max(data_sizes.values()) * args.n_train_epochs
            args.n_train_epochs = {d[0]: min(args.max_n_epochs, max_total_data_size//d[1]) for d in data_sizes.items()}
        else:
            args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}

    with open(args.label_map_file, 'r') as f:
        label_map = json.load(f)
    inverse_label_map = {v: k for k, v in label_map.items()}
    inverse_label_map[FILL_VAL] = "FILL"

    return args, model_config, model_class, tokenizer, config_class, special_token_ids, \
           special_tokens, data_attrs, tokens_weight, label_map, inverse_label_map


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = "{:.1f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format, filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

args, MODEL_CONFIG, MODEL_CLASS, TOKENIZER, CONFIG_CLASS, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, DATA_ATTRS, TOKENS_WEIGHT, LABEL_MAP, INVERSE_LABEL_MAP = parse_args()

if args.use_task_in_ner:
    for task in args.tasks:
        task_label_idx = len(LABEL_MAP)
        LABEL_MAP["__%s__" % task] = task_label_idx
        INVERSE_LABEL_MAP[task_label_idx] = "__%s__" % task

if args.ic:
    for l in LABEL_MAP:
        if l != "Other":
            TOKENIZER.add_tokens(["__%s__" % l])

if args.add_ent_tokens:
    ent_tokens = set()
    for l in LABEL_MAP:
        if l[0] != "O":
            ent_tokens.add(l[2:])
    TOKENIZER.add_tokens(["__%s__" % l for l in ent_tokens])

    start_label_idx = len(LABEL_MAP)
    LABEL_MAP["__start__"] = start_label_idx
    INVERSE_LABEL_MAP[start_label_idx] = "__start__"
    TOKENIZER.add_tokens(["__start__"])


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def get_gen_token(task):
    if args.add_task_tokens:
        return '__' + task + '__'
    else:
        return '__gen__'


def get_model_dir(tasks):
    return os.path.join(args.model_dir_root, tasks[0]) if args.seq_train_type != "multitask" else args.model_dir_root


for task in args.tasks:
    model_dir = get_model_dir([task])
    make_dir(model_dir)

    gen_token = get_gen_token(task)
    # TOKENIZER.add_tokens([gen_token, "pad_token"])
    TOKENIZER.add_tokens([gen_token])
    SPECIAL_TOKENS[task] = gen_token
    SPECIAL_TOKEN_IDS[task] = TOKENIZER.convert_tokens_to_ids(gen_token)
    # SPECIAL_TOKEN_IDS["pad_token"] = TOKENIZER.convert_tokens_to_ids("pad_token")
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[task]))
    # logger.info('pad token = {} , pad token id = {}'.format("pad_token", SPECIAL_TOKEN_IDS["pad_token"]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([len(TOKENIZER) - int(TOKENS_WEIGHT.shape[0])]).cuda()))

for task in args.tasks:
    model_dir = get_model_dir([task])
    TOKENIZER.save_pretrained(model_dir)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir, CONFIG_NAME))


TASK_DICT = {
    "squad1": {
               "train":os.path.join(args.data_dir,"squad-train-v1.1.json"),
               "eval":os.path.join(args.data_dir,"squad-dev-v1.1.json"),
               "test":os.path.join(args.data_dir,"squad-dev-v1.1.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "squad2": {
               "train":os.path.join(args.data_dir,"squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"squad-dev-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "iwslt.en.de": {
               "train":os.path.join(args.data_dir,"iwslt.en.de_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"iwslt.en.de_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"iwslt.en.de_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "cnn_dailymail": {
               "train":os.path.join(args.data_dir,"cnn_dailymail_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"cnn_dailymail_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"cnn_dailymail_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "multinli.in.out": {
               "train":os.path.join(args.data_dir,"multinli.in.out_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"multinli.in.out_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"multinli.in.out_to_squad-dev-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "sst": {
               "train":os.path.join(args.data_dir,"sst_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"sst_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"sst_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "srl": {
               "train":os.path.join(args.data_dir,"srl_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"srl_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"srl_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "zre": {
               "train":os.path.join(args.data_dir,"zre_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"zre_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"zre_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "woz.en": {
               "train":os.path.join(args.data_dir,"woz.en_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"woz.en_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"woz.en_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "wikisql": {
               "train":os.path.join(args.data_dir,"wikisql_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"wikisql_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"wikisql_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "schema": {
               "train":os.path.join(args.data_dir,"schema_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"schema_to_squad-dev-v2.0.json"),
               "test":os.path.join(args.data_dir,"schema_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "ag": {
               "train":os.path.join(args.data_dir,"ag_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"ag_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "dbpedia": {
               "train":os.path.join(args.data_dir,"dbpedia_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"dbpedia_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yahoo": {
               "train":os.path.join(args.data_dir,"yahoo_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yahoo_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "amazon": {
               "train":os.path.join(args.data_dir,"amazon_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"amazon_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "yelp": {
               "train":os.path.join(args.data_dir,"yelp_to_squad-train-v2.0.json"),
               "eval":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "test":os.path.join(args.data_dir,"yelp_to_squad-test-v2.0.json"),
               "n_train_epochs": args.n_train_epochs 
    },
    "conll_eng": {
               "train": os.path.join(args.data_dir, "conll_eng-train.json"),
               "eval": os.path.join(args.data_dir, "conll_eng-testa.json"),
               "test": os.path.join(args.data_dir, "conll_eng-testb.json"),
               "n_train_epochs": 25
    },
    "ask_ubuntu": {
               "train": os.path.join(args.data_dir, "ask_ubuntu-train.json"),
               "eval": os.path.join(args.data_dir, "ask_ubuntu-test.json"),
               "test": os.path.join(args.data_dir, "ask_ubuntu-test.json"),
               "n_train_epochs": 100
    },
    "wnut": {
               "train": os.path.join(args.data_dir, "wnut_train.json"),
               "eval": os.path.join(args.data_dir, "wnut_dev.json"),
               "test": os.path.join(args.data_dir, "wnut_test.json"),
               "n_train_epochs": 25
    },
    "wnut_O": {
               "train": os.path.join(args.data_dir, "wnut_train_O.json"),
               "eval": os.path.join(args.data_dir, "wnut_dev_O.json"),
               "test": os.path.join(args.data_dir, "wnut_test_O.json"),
               "n_train_epochs": 25
    },
    "so_1": {
               "train": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_train_1.json"),
               "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_1.json"),
               "test": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_1.json"),
               "n_train_epochs": 10
    },
    "so_2": {
               "train": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_train_2.json"),
               "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_2.json"),
               "test": os.path.join(args.data_dir, "so_data",  "skewed_splits", "so_test_2.json"),
               "n_train_epochs": 10
    },
    "so_3": {
               "train": os.path.join(args.data_dir, "so_data",  "skewed_splits", "so_train_3.json"),
               "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_3.json"),
               "test": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_3.json"),
               "n_train_epochs": 10
    },
    "so_4": {
               "train": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_train_4.json"),
               "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_4.json"),
               "test": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_4.json"),
               "n_train_epochs": 10
    },
    "so_5": {
               "train": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_train_5.json"),
               "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_5.json"),
               "test": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_5.json"),
               "n_train_epochs": 10
    },
    "so_all_1": {
               "train": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_train_all_1.json"),
               "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_1.json"),
               "test": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_1.json"),
               "n_train_epochs": 5
    },
    "so_all_2": {
                "train": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_train_all_2.json"),
                "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_2.json"),
                "test": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_2.json"),
                "n_train_epochs": 5
    },
    "so_all_3": {
                "train": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_train_all_3.json"),
                "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_3.json"),
                "test": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_3.json"),
                "n_train_epochs": 5
    },
    "so_all_4": {
                "train": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_train_all_4.json"),
                "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_4.json"),
                "test": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_4.json"),
                "n_train_epochs": 5
    },
    "so_all_5": {
                "train": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_train_all_5.json"),
                "eval": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_5.json"),
                "test": os.path.join(args.data_dir, "so_data", "skewed_splits", "so_test_5.json"),
                "n_train_epochs": 5
    },
    "so_t_1": {
               "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_1.json"),
               "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_1.json"),
               "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_1.json"),
               "n_train_epochs": 10
    },
    "so_t_2": {
               "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_2.json"),
               "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_2.json"),
               "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_2.json"),
               "n_train_epochs": 10
    },
    "so_t_3": {
               "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_3.json"),
               "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_3.json"),
               "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_3.json"),
               "n_train_epochs": 10
    },
    "so_t_4": {
               "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_4.json"),
               "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_4.json"),
               "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_4.json"),
               "n_train_epochs": 10
    },
    "so_t_5": {
               "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_5.json"),
               "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_5.json"),
               "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_5.json"),
               "n_train_epochs": 10
    },
    "so_t_all_1": {
               "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_all_1.json"),
               "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_1.json"),
               "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_1.json"),
               "n_train_epochs": 5
    },
    "so_t_all_2": {
                "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_all_2.json"),
                "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_2.json"),
                "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_2.json"),
                "n_train_epochs": 5
    },
    "so_t_all_3": {
                "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_all_3.json"),
                "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_3.json"),
                "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_3.json"),
                "n_train_epochs": 5
    },
    "so_t_all_4": {
                "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_all_4.json"),
                "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_4.json"),
                "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_4.json"),
                "n_train_epochs": 5
    },
    "so_t_all_5": {
                "train": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_train_all_5.json"),
                "eval": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_5.json"),
                "test": os.path.join(args.data_dir, "so_data", "temporal_splits", "so_temporal_test_5.json"),
                "n_train_epochs": 5
    },
}
