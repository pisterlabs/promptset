# python3
# Create date: 2023-10-19
# Author: Scc_hy
# Chapter: 10- Training Transformers from Scratch
# https://aliendao.cn/#/
# ====================================================================================

__doc__ = """
look at some aspects of training that we have not considered yet, such as the following:
- Gathering and processing a very large dataset
- Creating a custom tokenizer for our dataset
- Training a model on multiple GPUs at scale.

distributed training - Accelerate
"""
from transformers import pipeline, set_seed 
from datasets import load_dataset, DownloadConfig
import psutil 
import os
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import IterableDataset
from transformers import GPT2Config, GPT2Model
from transformers import OpenAIGPTConfig, OpenAIGPTModel
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb
os.environ['CURL_CA_BUNDLE'] = ''
# 1. Large Datasets and Where to Find Them
# -----------------------------------------------------

## 1.1. Challenges of Building a Large-Scale Corpus
# *****************************************************
some_info = """
- A significant proportion of the C4 corpus is machine-translated rather than translated by humans.
- Disparate erasure if African-American English as a result of stopword filtering in C4 resulted in an
underrepresentation of such content
- It is typically difficult in a large text corpus to find a middle ground between including (often too much)
sexually or other explicit content and totally erasing all mention of sexuality or gender.As a surprising 
consequence of this, a rather common word like "sex"(which can have both neutral and explicit meanings) is
completely unknown to a tokenizer that is trained on C4, since this word is fully absent from the corpus.
- There are many occurrences of copyright violation in BookCorpus, and probably in other large-scale datasets as well.
- There is genre skew toward "romance" novels in BookCorpus.

GPT was mostly trained on BookCorpus
GPT-2 was trained on web pages, blogs, and news articles linked from Reddit.
so that the main difference is the pretraining dataset,
"""
g_gpt2 = pipeline('text-generation', model='gpt2')
# g_gpt2 = GPT2Model(GPT2Config())
# g_gpt = OpenAIGPTModel(OpenAIGPTConfig())
g_gpt  = pipeline('text-generation', model='openai-gpt')



def model_size(model):
    return sum(t.numel() for t in model.parameters())

print(f"GPT size: {model_size(g_gpt.model)/1000**2:.1f}M parameters")
print(f"GPT2 size: {model_size(g_gpt2.model)/1000**2:.1f}M parameters")
# GPT size: 116.5M parameters
# GPT2 size: 124.4M parameters

def enum_pipeline_output(pipe, prompt, num_return_sequences):
    out = pipe(prompt, num_return_sequences=num_return_sequences, clean_up_tokenization_spaces=True)
    return "\n".join(f'{i+1}.' + s['generated_text'] for i, s in enumerate(out))

prompt = '\nWhen they came back'
print('GPT completions:\n' + enum_pipeline_output(g_gpt, prompt, 3), "\n")
print('GPT2 completions:\n' + enum_pipeline_output(g_gpt2, prompt, 3))

summary = """
In general, any model trained on a dataset will reflect the language bias and over- or underrepresentation of
populations and events in its training data.
"""

## 1.2 Building a Custom Code Dataset
# To simplify the task a bit, we’ll focus on building a code generation model for the Python programming language only
# *****************************************************
info = """
GitHub repositories can be accessed in two main ways:
- Via the GitHub REST API, like we saw in Chapter 9 when we download all the GitHub issues of the 
hugging face Transformers repository
- Via public dataset inventories like Google BigQuery
    - The bigquerypublic-data.github_repos.contents table contains copies of all ASCII files 
        that are less than 10 MB in size
        Projects also need to be open source to be included, as determined
        by GitHub’s License API.

Create a dataset with Google BigQuery
1. Create a Google cloud account
2. Create a Google BigQuery project under your account
3. In this project, create a dataset
4. In this dataset, create a table where the result of the SQL request will be stored 
5. Prepare and run the following SQL query on the github_repos.

two step:
1. Export your results to Google Cloud:
    a. Create a bucket and a folder in Google Cloud Storage (GCS).
    b. Export your table to this bucket by selecting Export > Export to GCS, with an export format of JSON and gzip compression.
2. To download the bucket to your machine, use the gsutil library.
    a. Install gsutil with `pip install gsutil`
    b. Configure gsutil with your Google account: gsutil config.
    c. Copy your bucket on your machine:
        $ gsutil -m -o "GSUtil:parallel_process_count=1" cp -r gs://<name_of_bucket>
        $ download direct: gir clone https://huggingface.co/datasets/transformersbook/codeparrot
"""

sql_ = """
-- This command processes about 2.6 TB of data to extract 26.8 million files.
-- The result is a dataset of about 50 GB of compressed JSON files, each containing the source code of Python files

SELECT f.repo_name, f.path, c.copies, c.size, c.content, l.license
FROM  `bigquery-public-data.github_repos.files` AS f
JOIN  `bigquery-public-data.github_repos.contents` AS c
ON    f.id = c.id
JOIN  `bigquery-public-data.github_repos.licenses` AS l
ON     f.repo_name = l.repo_name
WHERE   NOT c.binary
AND    ((f.path LIKE '%.py')
AND    (c.size BETWEEN 1024 AND 1048575))
;
"""





## 1.3 Working with Large Datasets
# *****************************************************
### Memory mapping

download_cfg = DownloadConfig(delete_extracted=True)
dataset = load_dataset('./codeparrot', split='train', download_config=download_cfg)

print(f'Number of python files code in dataset: {len(dataset)}')
ds_size = sum(os.stat(f['filename'] for f in dataset.cache_files))
print(f'Dataset size (cache file): {ds_size / 2**30:.2f} GB')
print(f'RAM used: {psutil.Process(os.getpid()).memory_info().rss >> 20} MB')
info = """
Number of python files code in dataset: 18695559
Dataset size (cache file): 183.68 GB
RAM memory used: 4924 MB

In addition, the zero-copy/zero-overhead format uses
Apache Arrow under the hood, which makes it very efficient to access any element.

Some datasets (reaching up to 1TB or more) will be difficult to fit even an a standard hard drive.
In this case, an alternative to scaling up the server your are using is to stream the dataset.
"""
#### Streaming
streamed_dataset = load_dataset('./codeparrot', split='train', streamimg=True)
iterator = iter(streamed_dataset)
print(dataset[0] == next(iterator))
print(dataset[1] == next(iterator))

# directly download samples without downloading the raw files locally
remote_dataset = load_dataset('transformersbook/codeparrot', split="train", streaming=True)



## 1.4 Adding Datasets to the Hugging Face Hub
# *****************************************************
steps = """
- Easily access it from our training server.
- See how streaming datasets work seamlessly with datasets from the Hub.
- Share it with the community, including you, dear reader


huggingface-cli login
huggingface-cli repo create --type dataset --organization transformersbook \
codeparrot-train
huggingface-cli repo create --type dataset --organization transformersbook \
codeparrot-valid
"""


# 2. Building a Tokenizer
# -----------------------------------------------------

info = """

"""


def tok_list(tokenizer, string):
    input_ids = tokenizer(string, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode(tok) for tok in input_ids]


tokenizer_T5 = AutoTokenizer.from_pretrained("t5-base")
tokenizer_camembert = AutoTokenizer.from_pretrained("camembert-base")
print(f'T5 tokens for "sex": {tok_list(tokenizer_T5,"sex")}')
print(f'CamemBERT tokens for "being": {tok_list(tokenizer_camembert,"being")}')
# T5 tokens for "sex": ['', 's', 'ex']
# CamemBERT tokens for "being": ['be', 'ing']


## 2.1 The Tokenizer Model
# *****************************************************
info = """
the tokenizer is a processing pipeline consisting of four steps:
1- normalization
2- pretokenization
3- the tokenizer model
4- postprocessing

there are several subwords tokenization algorithms that can be used, such as BPE, WordPiece and Unigram.

"""

## 2.2 Measuring Tokenizer Performance
# *****************************************************
info = """
- Subword fertility, which calculates the average number of subwords produced per tokenized word.
- Proportion of continued words, which refers to the proportion of tokenized words in a corpus that are split into at least two subtokens.
- Coverage metrics like the proportion of unknown words or rarely used tokens in a tokenized corpus.

generally best estimated by using the downstream performance of the model as the ultimate metric
"""
## 2.3 A Tokenizer for Python
# *****************************************************
py_code = r"""def say_hello():
    print("Hello, World!")
    
# Print it
say_hello();
"""
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print(tokenizer(py_code).tokens())
print(tokenizer.backend_tokenizer.normalizer)
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(py_code))
# [('def', (0, 3)), ('Ġsay', (3, 7)), ('_', (7, 8)), ('hello', (8, 13)), ('():',
# (13, 16)), ('ĊĠĠĠ', (16, 20)), ('Ġprint', (20, 26)), ('("', (26, 28)), ('Hello',
# (28, 33)), (',', (33, 34)), ('ĠWorld', (34, 40)), ('!")', (40, 43)), ('Ġ#', (43,
# 45)), ('ĠPrint', (45, 51)), ('Ġit', (51, 54)), ('Ċ', (54, 55)), ('Ċ', (55, 56)),
# ('say', (56, 59)), ('_', (59, 60)), ('hello', (60, 65)), ('()', (65, 67)), ('Ċ',
# (67, 68))]

a, e = u"a", u"€"
byte = ord(a.encode("utf-8"))
print(f'`{a}` is encoded as `{a.encode("utf-8")}` with a single byte: {byte}')
byte = [ord(chr(i)) for i in e.encode("utf-8")]
print(f'`{e}` is encoded as `{e.encode("utf-8")}` with three bytes: {byte}')
info = """
why work on a byte level ?

background: 
    - build our vocabulary from the 143,859 Unicode characters, but we would
also like to include words—i.e., combinations of Unicode characters.
This will make our model’s embedding layer very large because it comprises
one vector for each vocabulary token.
    - if we only use the 256 byte values as our vocabulary, the input
sequences will be segmented in many small pieces (each byte constituting the Unicode
characters), and as such our model will have to work on long inputs and spend
significant compute power on reconstructing Unicode characters from their separate
bytes, and then words from these characters. (ByT5 model release for a detailed study of this overhead)

A middle-ground solution:
construct a medium-sized vocabulary by extending
the 256-word vocabulary with the most common combinations of bytes.
(This is the approach taken by the BPE algorithm)

GPT-2 tokenizer first maps all the 256 input bytes to Unicode string that can easily be digested by the
standard BPE algorithms——that is,
    we will map our 256 elementary values to Unicode string that all correspond to standard printable Unicode characters.
    we have 256 single values at the end, forming our base
    vocabulary, and that these 256 values are correctly handled by our BPE algorithm


explain of `print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(py_code))`
    - Spaces, and in particular consecutive spaces, are conserved (for instance, the three spaces in 'ĊĠĠĠ').
    - Consecutive spaces are considered as a single word.
    - Each space preceding a word is attached to and considered a part of the subsequent
        word (e.g., in 'Ġsay').


The vocabulary of our GPT-2 tokenizer comprises 50,257 words:
    - The base vocabulary with the 256 values of bytes.
    - 50,000 additional tokens created by repeatedly merging the most commonly co-occurring tokens.
    - A special character added to the vocabulary to represent document boundaries.
"""
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v, k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())
print(f'Size of our base vocabulary: {len(base_vocab)}')
print(f'First element: `{base_vocab[0]}`, last element: `{base_vocab[-1]}`')
# Size of our base vocabulary: 256
# First element: `!`, last element: `Ń`


## 2.4 Training a Tokenizer
# *****************************************************
info = """
Retraining a tokenizer provided by huggingface transformers is simple:
- Specify our target vocabulary size.
- Prepare an iterator to supply lists of input string to process to train the tokenizer's  model.
- Call the train_new_form_iterator() method.

In a nutshell, the tokenizer is just trained to know which letter combinations are the most frequent in our corpus.


This makes sense since GPT-2 was trained on a corpus centered around Reddit.
"""
# hen looking at the longest words in the vocabulary of the GPT-2 tokenizer:
tokens = sorted(tokenizer.vocab.items(), key=lambda x: len(x[0]), reverse=True)
print([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[:8]]);
# ['ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ', '
# =================================================================', '
# ----------------------------------------------------------------
# ',
tokens = sorted(tokenizer.vocab.items(), key=lambda x: x[1], reverse=True)
print([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[:12]]);
# ['<|endoftext|>', ' gazed', ' informants', ' Collider', ' regress', 'ominated',
# ' amplification', 'Compar', '..."', ' (/', 'Commission', ' Hitman']

# '<|endoftext|>' was added after the BPE vocabulary was built.

# let’s select about 1–2 GB of data, or about 100,000 documents from our corpus:
length = 100000
dataset_name = 'transformersbook/codeparrot-train'
dataset = load_dataset(dataset_name, split="train", streaming=True)
iter_dataset = iter(dataset)

def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)['content'] for _ in range(batch_size) ]

new_tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(),
    vocab_size=12500,
    initial_alphabet=base_vocab
)
tokens = sorted(new_tokenizer.vocab.items(), key=lambda x: x[1], reverse=False)
print([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[257:280]])
# [' ', ' ', ' ', ' ', 'se', 'in', ' ', 're', 'on', 'te', '\n
# ', '\n ', 'or', 'st', 'de', '\n ', 'th', 'le', ' =', 'lf', 'self',
# 'me', 'al']
print([f'{new_tokenizer.convert_tokens_to_string(t)}' for t,_ in tokens[-12:]])
# [' capt', ' embedded', ' regarding', 'Bundle', '355', ' recv', ' dmp', ' vault',
# ' Mongo', ' possibly', 'implementation', 'Matches']
# Let’s check if all the Python reserved keywords are in the vocabulary:
import keyword
print(f'There are in total {len(keyword.kwlist)} Python keywords.')

for keyw in keyword.kwlist:
    if keyw not in new_tokenizer.vocab:
        print(f'No, keyword `{keyw}` is not in the vocabulary')

info = """
There are in total 35 Python keywords.
No, keyword `await` is not in the vocabulary
No, keyword `finally` is not in the vocabulary
No, keyword `nonlocal` is not in the vocabulary

It appears that several quite frequent keywords, like finally, are not in the vocabulary
either. Let’s try building a larger vocabulary using a larger sample of our dataset.
For instance, we can build a vocabulary of 32,768 words (multiples of 8 are better for
some efficient GPU/TPU computations)
"""
length = 200000
new_tokenizer_larger = tokenizer.train_new_from_iterator(batch_iterator(),
vocab_size=32768, initial_alphabet=base_vocab)

## 2.5 Saving a Custom Tokenizer on the Hub
# *****************************************************
model_ckpt = "codeparrot"
org = "transformersbook"
new_tokenizer_larger.push_to_hub(model_ckpt, organization=org)
reloaded_tokenizer = AutoTokenizer.from_pretrained(org + "/" + model_ckpt)


# 3. Training a Model from Scratch
# -----------------------------------------------------
# First, we need to decide which architecture is best suited for code autocompletion

## 3.1 A Tale of Pretraining Objects
# *****************************************************
info = """
1. Causal language modeling - decoder only
2. Masked language modeling - endcoder only  (denoising objective.)
3. Seq2Seq training - encode-decoder


Since we want to build a code autocompletion model, we'll select the first objective
and choose a GPT architecture for the task. So let's initialize a fresh GPT-2 model!
"""
## 3.2 Initializing the Model
# *****************************************************
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# download -> kaggle -> local
config = AutoConfig.from_pretrained('gpt2-xl', vocab_size=32768) 
model = AutoModelForCausalLM.from_config(AutoConfig)
print(f'GPT-2 (xl) size: {model_size(model)/1000**2:.1f}M parameters')
# GPT-2 (xl) size: 1529.6M parameters
config_small = AutoConfig.from_pretrained("gpt2", vocab_size=32768) # len(tokenizer))
model_small = AutoModelForCausalLM.from_config(config_small)
print(f'GPT-2 size: {model_size(model_small)/1000**2:.1f}M parameters')
#  GPT-2 size: 111.0M parameters

## 3.3 Implementing the Dataloader
# *****************************************************
info = """
To be able to train with maximal efficiency,  we will want to supply our model with sequences filling its context.
    - every batch: the token padding the length of the clip(token_len, min(batch_max_len, total_max_len), total_max_len)
    - this will render our training slightly less efficient and forcus us to take care of padding and masking padded token labels.
        - some tricks: we can tokenize several examples and then concatenate them, separated by the special end-of-sequence token,
            to get a very long sequence.
            we split this sequence into equally sized chuncks.

input_characters = number_of_sequences * sequence_length * characters_per_token
    - number_of_sequences: is the number of (truncated) sequences we would like from our tokenizer
        - set number_of_sequences by computing the loss rate of our dataset which suggest to below 1%
    - sequence_length: is the number of tokens per sequence returned by the tokenizer
"""
### estimate the average character length per token in our dataset.
examples, total_characters, total_tokens = 500, 0, 0
dataset = load_dataset('transformersbook/codeparrot-train', split='train', streaming=True)

for _, example in tqdm(zip(range(examples), iter(dataset)), total=examples):
    total_characters += len(example['content'])
    total_tokens += len(tokenizer(example['content']).tokens())


characters_per_token = total_characters / total_tokens
print(characters_per_token) # 3.623


class ConstantLengthDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, seq_length=1024, num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
    
    def __iter__(self):
        iter_ = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer_, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m = f'Buffer full: {buffer_len} >= {self.input_characters}'
                    print(m)
                    break
                try:
                    m = f'Fill buffer: {buffer_len} < {self.input_characters}'
                    print(m)
                    buffer_.append(next(iter_)['content'])
                    buffer_len += 1
                except StopIteration:
                    iter_ = iter(self.dataset)

            all_tokens_ids = []
            tokenized_inputs = self.tokenizer(buffer_, truncation=False)
            for token_ipt in tokenized_inputs['inputs_ids']:
                all_tokens_ids.extend(token_ipt + [self.concat_token_id])
            
            for i in range(0, len(all_tokens_ids), self.seq_length):
                input_ids = all_tokens_ids[i:i+self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)
        
        
# dont need mask with the same length of return tokenize
shuffled_dataset = dataset.shuffle(buffer_size=100)
constant_length_dataset = ConstantLengthDataset(tokenizer, shuffled_dataset, num_of_sequences=10)
data_iter = iter(constant_length_dataset)
len_ = [len(b) for _, b in zip(range(5), data_iter)]
print(f"Lengths of the sequences: {len_}")
# Lengths of the sequences: [1024, 1024, 1024, 1024, 1024]


## 3.4 Defining the Training loop
# *****************************************************
info = """
data parallelism: which will help us utilize several GPUs for training
use huggingface Accelerate to make our code scalable.
- distributed training
- changing the underlying hardware for training —— easy
- make training scripts run with mixed precision and in any kind of distributed setting
"""
torch.cuda.get_device_name() # 'NVIDIA GeForce RTX 4070 Ti'
torch.cuda.device_count() # 1

# train loop -> chaper10_trainloop.py
# 1- config
config = {
    "train_batch_size": 2, # 12
    "valid_batch_size": 2, # 12
    "weight_decay": 0.1,
    "shuffle_buffer": 1000,
    "learning_rate": 2e-4, # 5e-4
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 750, # 2000
    "gradient_accumulation_steps": 16, # 1
    "max_train_steps": 50000, # 150000
    "max_eval_steps": -1,
    "seq_length": 1024,
    "seed": 1,
    "save_checkpoint_steps": 50000
}
args = Namespace(**config)

# ********************************************************
# the heart of the training script:
from accelerate import Accelerator
from Chapyter10_trainLoop import (
    setup_logging, create_dataloaders, 
    get_grouped_params, log_metrics,
    evaluate
)
from torch.optim import AdamW
from transformers.optimization import get_scheduler

set_seed(args.seed)
# Accelerator
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size

# Logging
logger, tb_writer, run_name = setup_logging(project_name.split("/")[1])
logger.info(accelerator.state)

# Load model and tokenizer
if accelerator.is_main_process:
    hf_repo = Repository('./', clone_from=project_name, revision=run_name)

model = AutoModelForCausalLM.from_pretrained("./", gradient_checkpointing=True)
tokenizer = AutoTokenizer.from_pretrained("./")

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(dataset_name)

# Prepare the optimizer and learning rate scheduler
opt = AdamW(get_grouped_params(model), lr=args.learning_rate)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=opt,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps
)

def get_lr():
    return opt.param_groups[0]['lr']

# Train model
model.train()
completed_steps = 0
for step, batch in enumerate(train_dataloader, start=1):
    loss = model(batch, labels=batch).loss
    log_metrics(
        step, {
            'lr': get_lr, 
            'samples': step * samples_per_step,
            'steps': completed_steps,
            'loss/train': loss.items()
            }
    )
    loss = loss / args.gradient_accumlation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        opt.step()
        lr_scheduler.step()
        opt.zero_grad()
        completed_steps += 1
    if step % args.save_checkpoint_steps == 0:
        logger.info('Evaluating and saving model checkpoint')
        eval_loss, perplexity = evaluate()
        log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained("./")
            hf_repo.push_to_hub(commit_message=f'step {step}')
    model.train()
    if completed_steps >= args.max_train_steps:
        break

# Evaluate and save the last checkpoint
logger.info('Evaluating and save model after training')
eval_loss, perplexity = evaluate()
log_metrics(step, {'loss/val': eval_loss, 'perplexity': perplexity})
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unwrapped_model.save_pretrained("./")
    hf_repo.push_to_hub(commit_message=f'final model')

important_info = """
1. Model saving

2. Optimization

3. Evaluation

4. Gradient accumulation and checkpointing


"""

