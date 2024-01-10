import json
import numpy as np
from collections import defaultdict
import ray 
import pandas as pd
import openai
# Import the tokenizer
from transformers import LlamaTokenizerFast
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

def check_data_for_format_errors(items: list):

    for line_num, batch in enumerate(items):
        prefix = f"Error in line #{line_num + 1}: "
        if not isinstance(batch, dict):
            raise DataFormatError(
                f"{prefix}Each line in the provided data should be a dictionary"
            )

        if "messages" not in batch:
            raise DataFormatError(
                f"{prefix}Each line in the provided data should have a 'messages' key"
            )

        if not isinstance(batch["messages"], list):
            raise DataFormatError(
                f"{prefix}Each line in the provided data should have a 'messages' key with a list of messages"
            )

        messages = batch["messages"]
        if not any(message.get("role", None) == "assistant" for message in messages):
            raise DataFormatError(
                f"{prefix}Each message list should have at least one message with role 'assistant'"
            )

        for message_num, message in enumerate(messages):
            prefix = f"Error in line #{line_num + 1}, message #{message_num + 1}: "
            if "role" not in message or "content" not in message:
                raise DataFormatError(
                    f"{prefix}Each message should have a 'role' and 'content' key"
                )

            if any(k not in ("role", "content", "name") for k in message):
                raise DataFormatError(
                    f"{prefix}Each message should only have 'role', 'content', and 'name' keys, any other key is not allowed"
                )

            if message.get("role", None) not in ("system", "user", "assistant"):
                raise DataFormatError(
                    f"{prefix}Each message should have a valid role (system, user, or assistant)"
                )


# Utility function for proper formatting of the data
def convert_message_list_to_text(messages: list) -> str:
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    text = ""

    if messages[0]["role"] == "system":
        messages = [
            {
                "role": messages[1]["role"],
                "content": B_SYS
                + messages[0]["content"]
                + E_SYS
                + messages[1]["content"],
            }
        ] + messages[2:]

    assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
            [msg["role"] == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )

    texts = []
    for prompt, answer in zip(messages[::2], messages[1::2]):
        texts.append(f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ")

    text = "</s><s>".join(texts)
    # add the bos and eos token at the beginning of the first turn and the end of the last turn
    text = "<s>" + text + " </s>"
    # During training last message should be from assistant (not from a user)
    assert (
        messages[-1]["role"] == "assistant"
    ), f"Last message must be from assistant, got {messages[-1]['role']}"

    return text


# Utility functions for calculating the statistics of the number of tokens in the dataset
def print_token_statistics(stats) -> None:
    for key in stats:
        print(f"Statistics for {key}:")
        if isinstance(stats[key], dict):
            for stat_key, stat_value in stats[key].items():
                print(f"\t{stat_key}: {stat_value:.3f}")
        else:
            print(f"\t{stats[key]}")
        print("")

def get_tokenized_stats(items: list, print_stats: bool = True):

    counters = defaultdict(list)
    for i, batch in enumerate(items):
        #print(i)
        messages = batch["messages"]

        # add message count
        counters["message"].append(len(messages))

        # add the number of tokens of this message to the token counter
        text = convert_message_list_to_text(messages)
        tokens = tokenizer(text)['input_ids']
        counters["token"].append(len(tokens))

    stats = {}
    for key, value in counters.items():
        stats[key] = {
            "max": float(np.max(value)),
            "min": float(np.min(value)),
            "median": float(np.median(value)),
            "mean": float(np.mean(value)),
            "p95": float(np.percentile(value, 95)),
            "p5": float(np.percentile(value, 5)),
        }
    stats["ds_size"] = len(items)

    if print_stats:
        print_token_statistics(stats)

    return stats

def batched_convert_messages_to_text(batch: pd.DataFrame) -> pd.DataFrame:
    """Converts a batch of messages (list of roles + content) to plain text."""
    df = []
    for _, b in batch.iterrows():
        text = convert_message_list_to_text(list(b["messages"]))
        df.append({"input": text})

    return pd.DataFrame(df)


LLAMA2_MODEL_SIZE = "7b" # "7b" or "13b"
DS_MAX_SIZE_LIMITS = {
    "7b": {
        512: 450_000,
        1024: 150_000,
        2048: 75_000,
        4096: 30_000
    },
    "13b": {
        512: 270_000,
        1024: 96_000,
        2048: 45_000,
        4096: 15_000
    },
}

def finetune_check(filename):
    # Load the dataset
    with open(filename, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    class DataFormatError(Exception):
        pass
    ## Format validate check
    try:
        check_data_for_format_errors(items)
        print("Data format is valid!")
    except DataFormatError as e:
        print("Data format is NOT valid!")
        print(e)
        
    ## Token length check    
    SUPPORTED_CONTEXT_LENGTHS = [512, 1024, 2048, 4096]

    tokenizer.pad_token = tokenizer.eos_token        
    stats = get_tokenized_stats(items, print_stats=True)
    for ctx_length in SUPPORTED_CONTEXT_LENGTHS:
        if ctx_length > stats["token"]["p95"]:
            break

    print("Automatically selected context length: ", ctx_length)     
    
    CONTEXT_LENGTH = ctx_length
    def collate_fn(batch: dict):
        return tokenizer(
            list(batch["input"]),
            padding="longest",
            max_length=CONTEXT_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
    
    ds_max_size = DS_MAX_SIZE_LIMITS[LLAMA2_MODEL_SIZE][CONTEXT_LENGTH]
    if len(items) > ds_max_size:
        raise ValueError(
            f"Dataset size ({len(items)}) exceeds the maximum allowable size ({ds_max_size})"
        )
        
    ### Token count
    #You can change the batch size per device here
    BSIZE_PER_DEVICE = 16

    # Creating a ray dataset for easier processing
    df = pd.DataFrame.from_dict(items)
    ds = ray.data.from_pandas(df)
    
    # Data preprocssing pipeline
    flattened_ds = ds.map_batches(
        batched_convert_messages_to_text, batch_size=16, batch_format="pandas"
    )

    data_set_tokens_per_epoch = 0
    trained_tokens_per_epoch = 0
    for batch in flattened_ds.iter_torch_batches(
        batch_size=BSIZE_PER_DEVICE, collate_fn=collate_fn
    ):
        trained_tokens_per_epoch += batch["input_ids"].numel()
        data_set_tokens_per_epoch += batch["attention_mask"].sum().item()

    print("Num tokens in dataset per epoch: ", data_set_tokens_per_epoch)
    print("Num tokens trained per epoch: ", trained_tokens_per_epoch)
    print("Padding inflation ratio: ", trained_tokens_per_epoch / data_set_tokens_per_epoch)
    
def finetune_run(train_fn, val_fn=None, token="esecret_", model="meta-llama/Llama-2-7b-chat-hf", suffix="finetune"): 
    
    openai.api_base = "https://api.endpoints.anyscale.com/v1"
    openai.api_key = token

    file = openai.File.create(
      file=open(train_fn, "rb"),
      purpose="fine-tune",
      user_provided_filename=train_fn,
    )
    if val_fn:
        file = openai.File.create(
          file=open(val_fn, "rb"),
          purpose="fine-tune",
          user_provided_filename=val_fn,
        )
    files = openai.File.list()
    train_id = val_id = None
    for file in files['data']:
        if file['filename'] == train_fn:
            train_id = file['id']
            continue
        if file['filename'] == val_fn:
            val_id = file['id']
            continue

    return openai.FineTuningJob.create(
        model=model,
        training_file=train_id,
        validation_file=val_id,
        suffix=suffix
    )
