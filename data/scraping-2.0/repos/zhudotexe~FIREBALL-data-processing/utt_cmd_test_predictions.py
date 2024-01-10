import json
import logging
import pathlib
import time

import openai

import prompts
from dataset.utils import read_jsonl_file

FULL_MODEL = "davinci:ft-ccb-lab-members-2022-11-18-00-38-05"
ABLATION_MODEL = "davinci:ft-ccb-lab-members-2022-11-18-01-21-00"
TEST_DATA_FILE = pathlib.Path("extract/ft-utt-cmd-test-1000.jsonl")
MODEL_NAME = "full"  # CHANGE ME
RESULT_FILE = pathlib.Path(f"utt-cmd-test-predictions-{MODEL_NAME}.jsonl")

log = logging.getLogger("predictions")


def get_prediction(prompt, model=FULL_MODEL):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0.2,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n<|aeot|>"],
    )
    return response["choices"][0]["text"].strip()


def main():
    # backup existing result file
    now = int(time.time())
    if RESULT_FILE.is_file():
        RESULT_FILE.rename(f"utt-cmd-test-predictions-{MODEL_NAME}-backup-{now}.jsonl")

    test_data = read_jsonl_file(TEST_DATA_FILE)
    out = open(RESULT_FILE, "w")
    for datum in test_data:
        prompt = prompts.utt_cmd_prompt(datum)  # CHANGE ME FOR ABLATIONS
        if len(prompt) > 8000:
            log.warning(f"Prompt is too long ({len(prompt)}) is probably more than 2048 tokens")
            continue
        prediction = get_prediction(prompt)
        log.info(prompt)
        log.info(prediction)
        datum[f"prediction_{MODEL_NAME}"] = prediction
        out.write(json.dumps(datum))
        out.write("\n")
        time.sleep(0.1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
