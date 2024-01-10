import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from copy import deepcopy

from tqdm import tqdm
import openai

from writing.data.jsonl import load_jsonl, save_jsonl, fpprint
from writing.utils.gpt import GPTAgent
from writing.utils.prompt import StoryGenerateTemplate

openai.api_key = os.environ.get("OPENAI_API_KEY")

# FIXME: Change this part to run the intended experiment
# ----------------------------
template = StoryGenerateTemplate
var_names = ["moments"]

DATA_PATH = "writing/data/story_test.jsonl"
TEST_NAME = "story1"
OUTPUT_JSONL_PATH = f"results/{TEST_NAME}.jsonl"
OUTPUT_TXT_PATH = f"results/{TEST_NAME}.txt"
# ----------------------------

agent = GPTAgent()
results = []

DUPLICATES = 1


def run():
    dataset = load_jsonl(DATA_PATH)

    for entry in tqdm(dataset):
        kwargs = {var_name: entry[var_name] for var_name in var_names}
        prompt = template.get_prompt(**kwargs)
        print(prompt)

        agent.reset_messages()
        agent.add_message(prompt)

        try:
            for _ in range(DUPLICATES):
                reply = agent.get_answer(timeout=20, max_trial=10)
                print(reply)

                entry_copy = deepcopy(entry)
                entry_copy["reply"] = reply
                results.append(entry_copy)

        except GPTAgent.GPTError:
            print("GPT call failed")


def save():
    save_jsonl(OUTPUT_JSONL_PATH, results)
    fpprint(OUTPUT_TXT_PATH, results, width=50, iterate=True)


def main():
    try:
        run()
    except KeyboardInterrupt:
        ...
    finally:
        save()


if __name__ == "__main__":
    main()
