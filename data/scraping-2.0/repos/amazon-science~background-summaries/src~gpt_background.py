import argparse
import json
import logging
import re
import time
from pathlib import Path

import openai
import pyhocon
import spacy
from transformers import GPT2TokenizerFast

from dataset import TSDataset

logger = logging.getLogger(__name__)

spacy_nlp = spacy.load("en_core_web_lg")


def get_chunks(data: list, bsz: int):
    for i in range(0, len(data), bsz):
        yield data[i : i + bsz]


def truncate_input(doc: str, tokenizer, max_length: int) -> list[str]:
    truncated_doc = doc
    tokens = tokenizer(truncated_doc)["input_ids"]
    while len(tokens) > max_length:
        truncated_doc = "".join(
            re.split(r"(Date: [0-9]{4}-[0-9]{2}-[0-9]{2})", truncated_doc, 2)[3:]
        )
        tokens = tokenizer(truncated_doc)["input_ids"]
    return truncated_doc


def write_log(messages: list, file_path: Path):
    past_messages = []
    if file_path.is_file():
        with open(file_path, "r") as rf:
            past_messages = json.load(rf)
    with open(file_path, "w") as wf:
        json.dump(past_messages + messages, wf, indent=2)


def summarize(
    model: str,
    temperature: float,
    src: list[str],
    bs: int,
    messages_path: Path,
):
    """generate summaries using GPT
    batching is just for writing message log

    Args:
        model (str): _description_
        temperature (float): _description_
        src (list[str]): _description_
        bs (int): _description_
        messages_path (Path): _description_

    Returns:
        _type_: _description_
    """
    prompt_token_count, completion_token_count = 0, 0
    preds = []

    logger.info(f"# inputs: {len(src)}")

    for idx, batch in enumerate(get_chunks(src, bs)):
        batch_preds, batch_log = [], []
        for doc in batch:
            messages = [
                {
                    "role": "user",
                    "content": doc,
                },
            ]
            batch_log += messages
            completion = None
            while completion is None:
                try:
                    completion = openai.ChatCompletion.create(
                        model=model,
                        temperature=temperature,
                        messages=messages,
                    )
                except Exception as E:
                    SLEEP = 300
                    logger.warning(f"found exception: {E}")
                    logger.warning(f"sleeping for {SLEEP} seconds")
                    time.sleep(SLEEP)

            batch_log += [completion["choices"][0]["message"]]
            batch_preds += [completion["choices"][0]["message"]["content"].strip("\n")]
            prompt_token_count += completion["usage"]["prompt_tokens"]
            completion_token_count += completion["usage"]["completion_tokens"]

        preds += batch_preds
        logger.info(
            f"batch: {idx+1:>5}"
            f" | prompt: {prompt_token_count:>10}"
            f" | completion: {completion_token_count:>10}"
            f" | cost: $"
            f"{(prompt_token_count+completion_token_count)*0.002/1000:.4f}"
        )
        write_log(batch_log, file_path=messages_path)

    return preds


def concat(src: list[str], config, z: list[str] = None):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    out = []
    for src_idx in range(len(src)):
        max_len = config["max_src_len"]

        # src prefix
        if config.get("src_prefix", False):
            prefix_toks = tokenizer(
                config["src_prefix"], padding=False, truncation=False
            )
            max_len -= len(prefix_toks["input_ids"])

        # z prefix (query/guidance)
        if config.get("z_prefix", False):
            prefix_toks = tokenizer(config["z_prefix"], padding=False, truncation=False)
            max_len -= len(prefix_toks["input_ids"])

        # task suffix
        assert "task_suffix" in config
        prefix_toks = tokenizer(config["task_suffix"], padding=False, truncation=False)
        max_len -= len(prefix_toks["input_ids"])

        # truncate docs to `max_len`
        truncated_src = truncate_input(
            src[src_idx], tokenizer=tokenizer, max_length=max_len
        )

        if z is not None:
            out_src = (
                config["z_prefix"]
                + z[src_idx]
                + config["src_prefix"]
                + truncated_src
                + config["task_suffix"]
            )
        else:
            out_src = truncated_src + config["task_suffix"]
        out += [out_src]

    return out


def prepare_inputs(updates: list[str], ts: list[str], config):
    # timestamp each update
    src = [f"Date: {ts[idy]}, Article: {updates[idy]}" for idy in range(len(updates))]
    # group updates until current timestamp
    src = [" ".join(src[:idy]) for idy in range(1, len(src))]

    # add prefix/suffix and truncate documents
    z = None
    if "z_type" in config:
        logger.info(f"using {config['z_type']} as Z")
        if config["z_type"] == "update":
            z = updates[1:]
        elif config["z_type"] == "entities":
            ents = []
            for doc in spacy_nlp.pipe(updates[1:], n_process=1):
                ents += [", ".join([ent.text for ent in doc.ents]) + "."]
            z = ents

    src = concat(src=src, config=config, z=z)
    return src


def main(config):
    # load dataset
    ts_data = TSDataset(config["dataset_path"] / "events")
    event2data = ts_data.get_summaries()

    # load split
    split_path = config["dataset_path"] / "splits" / f"{config['split']}.txt"
    with open(split_path, "r") as rf:
        split_events = [line.strip() for line in rf.readlines()]

    for event in split_events:
        ts, updates, _ = event2data[event]

        # path to write prediction tsv files
        event_pred_path = config["output_path"] / "preds" / event
        event_pred_path.mkdir(exist_ok=True, parents=True)
        # path to write GPT message log
        log_path = config["output_path"] / "log"
        log_path.mkdir(exist_ok=True, parents=True)

        ann_updates = list(zip(*updates))  # grouped by annotators
        assert len(ann_updates) == 3, len(ann_updates)

        for idx in range(len(ann_updates)):
            # predict background summaries per annotator

            # output path
            logger.info(f"{event}-annotator-{idx+1}")
            ann_out_path = event_pred_path / f"annotator{idx+1}.tsv"
            if ann_out_path.is_file():
                logger.info("prediction tsv exists! skipping")
                continue

            # files to write gpt logs
            messages_path = log_path / f"{event}-annotator{idx+1}-messages.json"
            messages_path.unlink(missing_ok=True)

            src = prepare_inputs(updates=ann_updates[idx], ts=ts, config=config)
            preds = summarize(
                model=config["model"],
                temperature=config["temperature"],
                src=src,
                bs=16,
                messages_path=messages_path,
            )
            assert len(preds) == len(ann_updates[idx]) - 1
            preds = [""] + preds  # add empty background for first timestep

            with open(event_pred_path / f"annotator{idx+1}.tsv", "w") as wf:
                wf.write("Date\tUpdate\tBackground\n")
                for _ts, _update, _background in zip(ts, ann_updates[idx], preds):
                    wf.write(f"{_ts}\t{_update}\t{_background}\n")


def init_config(config_path: Path, config_name: str, debug: bool = False):
    config = pyhocon.ConfigFactory.parse_file(config_path)[config_name]
    print(config)

    for x in ["dataset_path", "output_path", "log_path"]:
        config[x] = Path(config[x])

    config["output_path"] /= f"{config_name}"
    config["log_path"] /= f"{config_name}"

    config["output_path"].mkdir(exist_ok=True, parents=True)
    config["log_path"].mkdir(exist_ok=True, parents=True)

    LOG_LEVEL = logging.DEBUG if debug else logging.INFO
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            config["log_path"] / "log.txt",
            mode="w",
        ),
    ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=LOG_LEVEL,
        handlers=handlers,
    )

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    return config


def parse_args():
    parser = argparse.ArgumentParser("predict background summaries using GPT-3.5")
    parser.add_argument("--config", type=Path, help="config path")
    parser.add_argument("--config-name", type=str, help="config name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = init_config(args.config, args.config_name)
    main(config)
