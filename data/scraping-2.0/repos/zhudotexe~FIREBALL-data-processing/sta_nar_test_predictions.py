import json
import logging
import pathlib
import time

import openai

import prompts
from dataset.utils import read_jsonl_file

FULL_MODEL = "davinci:ft-ccb-lab-members-2022-12-01-02-31-41"
STATE_ABLATION_MODEL = "davinci:ft-ccb-lab-members-2022-12-01-03-28-31"
COMMAND_ONLY_MODEL = "davinci:ft-ccb-lab-members-2022-12-02-00-34-21"
DIALOG_CONTINUATION_MODEL = "davinci:ft-ccb-lab-members-2022-12-02-21-10-07"
TEST_DATA_FILE = pathlib.Path("extract/ft-sta-nar-test-1000.jsonl")
MODEL_NAME = "full"  # CHANGE ME
RESULT_FILE = pathlib.Path(f"sta-nar-test-predictions-{MODEL_NAME}.jsonl")

log = logging.getLogger("predictions")


def get_prediction(prompt, model=FULL_MODEL):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0.7,
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
        RESULT_FILE.rename(f"sta-nar-test-predictions-{MODEL_NAME}-backup-{now}.jsonl")

    test_data = read_jsonl_file(TEST_DATA_FILE)
    out = open(RESULT_FILE, "w")
    for datum in test_data:
        prompt = prompts.sta_nar_prompt(datum)  # CHANGE ME FOR ABLATIONS
        # prompt = prompts.sta_nar_dialog_continuation_prompt(datum)
        try:
            prediction = get_prediction(prompt, model=FULL_MODEL)  # CHANGE ME FOR ABLATIONS TOO
        except openai.InvalidRequestError:
            log.exception("Failed to get prediction:")
            continue
        log.info(prompt)
        log.info(prediction)
        datum[f"prediction_{MODEL_NAME}"] = prediction
        out.write(json.dumps(datum))
        out.write("\n")
        time.sleep(0.1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
