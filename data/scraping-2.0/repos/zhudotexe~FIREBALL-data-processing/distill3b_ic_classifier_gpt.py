"""
Distill3: Given triples, filter the events in the triples:
- remove OOC utterances (anything not classified as in-character with >80% confidence)

Input: {"before": [Message...], "commands": [Event...], "after": [Message...]}

Output: {
    "before": [Message...],
    "commands": [Event...],
    "after": [Message...],
}
- with `after` filtered to only include IC-classified utterances (maybe `before` too - see how you feel about it)
"""
import glob
import logging
import pathlib
import time

import openai
import tqdm.contrib.concurrent
import tqdm.contrib.logging

from dataset.utils import read_gzipped_file, write_jsonl

DATA_DIR = pathlib.Path("data/")
# IN_DIR = pathlib.Path("extract/experiment2/")
IN_DIR = pathlib.Path("extract/experiment3a/")
OUT_DIR = pathlib.Path("extract/experiment3b/")

CLASSIFIER_FINETUNE = "ada:ft-ccb-lab-members-2022-11-28-18-29-25"

log = logging.getLogger("distill3")
loglevel = logging.INFO
logging.getLogger("openai").setLevel(logging.WARNING)


def get_ooc_ic_label(text, finetuned_model=CLASSIFIER_FINETUNE):
    if not text:
        return "out-of-character", 1
    if "OOC" in text or "OOG" in text or text.startswith("("):
        return "out-of-character", 1
    #  if text.startswith('"'):
    #  	return "in-character"
    if len(text.split(" ")) > 200:
        text = " ".join(text.split(" ")[:200])
    for _ in range(3):
        response = openai.Completion.create(
            model=finetuned_model,
            prompt=text + "\nlabel: ",
            temperature=0,
            max_tokens=7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["###", "\n"],
            logprobs=1,
        )
        time.sleep(0.05)
        label = response["choices"][0]["text"].strip()
        if label == "in-character" or label == "out-of-character" or label == "mixed":
            prob = 2 ** (response["choices"][0]["logprobs"]["token_logprobs"][0])
            return label, prob
    return None, 1


def process_triple(triple) -> dict | None:
    after = triple["after"]
    filtered_utterances = []
    for event in after:
        content = event["content"].strip()
        label, prob = get_ooc_ic_label(content)
        log.info(f"{content}\n---\n{label} {prob:.2%}\n=====\n")
        if not (label == "in-character" and prob > 0.8):
            continue
        filtered_utterances.append(event)
    triple["after"] = filtered_utterances
    log.info(f'after content: {sum(len(msg["content"]) for msg in triple["after"])} in {len(triple["after"])} events')
    if triple["after"] or triple["before"]:
        return triple
    return None


def process_file(fp: pathlib.Path):
    """
    Given a path to a file containing a list of triples, filter the triples and return a pair of
    (n_triples_in, n_triples_out).
    """
    triple_stream = read_gzipped_file(fp)
    num_triples_in = 0
    combat_id, *_ = fp.stem.split(".")
    out = []

    for triple in triple_stream:
        num_triples_in += 1
        processed = process_triple(triple)
        if processed is not None:
            out.append(processed)

    # discard if we have nothing
    if not out:
        log.info("nothing was processed")
        return num_triples_in, 0

    # see what we get
    write_jsonl(OUT_DIR / f"{combat_id}.jsonl.gz", out)
    return num_triples_in, len(out)


if __name__ == "__main__":
    logging.basicConfig(level=loglevel, format="%(name)s:%(levelname)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filenames = sorted(glob.glob("*.gz", root_dir=IN_DIR))
    files = [pathlib.Path(IN_DIR, fn) for fn in filenames]
    with tqdm.contrib.logging.logging_redirect_tqdm():
        results = []
        for d in tqdm.tqdm(files):
            results.append(process_file(d))

    kept_distill_count = sum(1 for (i, o) in results if o)
    n_triples_in = sum(i for i, o in results)
    n_triples_out = sum(o for i, o in results)
    print(
        f"Distill complete!\n"
        f"Instances: {len(filenames)} -> {kept_distill_count}\n"
        f"Triples: {n_triples_in} -> {n_triples_out}"
    )
