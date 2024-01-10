from copy import deepcopy

from tqdm import tqdm
import openai

from writing.data.jsonl import load_jsonl, save_jsonl, fpprint
from writing.utils.gpt import GPTAgent
from writing.utils.prompt import MomentReplyTemplate


openai.api_key = ""  # FIXME: API key here

DATA_PATH = "writing/data/reply_test.jsonl"
TEST_NAME = "reply1"
OUTPUT_JSONL_PATH = f"results/{TEST_NAME}.jsonl"
OUTPUT_TXT_PATH = f"results/{TEST_NAME}.txt"

agent = GPTAgent()
results = []

DUPLICATES = 2


def run():
    dataset = load_jsonl(DATA_PATH)

    for entry in tqdm(dataset):
        prompt = MomentReplyTemplate.get_prompt(moment=entry["moment"])
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
