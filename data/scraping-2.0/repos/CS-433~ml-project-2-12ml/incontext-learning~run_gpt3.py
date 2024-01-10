# Contains utils for running GPT-3 experiments.
# The notebook is more suitable for running as it can be interrupted (by the user or by exceptions)
# and the progress is saved.

import datasets
import time
import numpy as np
import zipfile
import pandas as pd
import pickle
import openai
import argparse


curr_prompt_idx = 0 # for interacting with OpenAI API


class Prompter:
    """Convenience class for constructing prompts"""

    def __init__(self, train_set, k_shot=0, explain=False):
        self.conjunction = {
            "effect": ", therefore",
            "cause": " because"
        }
        self.label_map = {
            0: "(a)",
            1: "(b)"
        }
        self.train_set = train_set
        self.k_shot = k_shot
        self.explain = explain

    def construct_instance(self, datapoint, give_answer=False, prepend=False):
        """Constructs a single question-answer instance."""

        premise = self.convert_premise(datapoint["premise"])
        qa_instance = ""
        if prepend:
            qa_instance += f"Instruction: for each question, {'provide a one-sentence explanation before giving' if self.explain else 'give'} the correct option (a) or (b).\n\n"
        qa_instance += f"""Question: {premise}{self.conjunction[datapoint["question"]]}
(a) {self.convert_choice(datapoint["choice1"])}
(b) {self.convert_choice(datapoint["choice2"])}"""
        qa_instance += "\nAnswer:"
        if give_answer:
            if self.explain:
                qa_instance += ' ' + datapoint['conceptual_explanation']
                qa_instance += f' So the answer is {self.label_map[datapoint["label"]]}.'
            else:
                qa_instance += f' {self.label_map[datapoint["label"]]}'
        return qa_instance

    def get_k_train_examples(self):
        """Generates k few-shot examples"""

        i = np.random.randint(0, 100, self.k_shot).tolist()
        d = self.train_set[i]
        d = [dict(zip(d, col)) for col in zip(*d.values())]
        return [self.construct_instance(example, give_answer=True, prepend=i==0) for i, example in enumerate(d)]

    def make_prompt(self, datapoint):
        """Makes a single prompt from a datapoint"""

        train_examples = self.get_k_train_examples()
        query = self.construct_instance(datapoint)
        prompt = "" if self.k_shot > 0 else "Instruction: for each question, give the correct option (a) or (b).\n\n"
        for train_example in train_examples:
            prompt += train_example
            prompt += "\n\n"
        prompt += query
        return {"prompt": prompt}

    def convert_choice(self, choice):
        """De-capitalizes the first character of the sentence"""
        return choice[0].lower() + choice[1:]

    def convert_premise(self, premise):
        """Removes the full-stop at the end of the sentence"""
        return premise.strip()[:-1]

    def get_prompt_skeleton(self):
        pass


def get_gpt3_prediction(prompt):
    """Makes a single call to the API and retrieves the response.
    Temperature: higher value means more diverse generated text.
    We do want more diverse generated causal explanations"""

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256
    )
    return response.choices[0].text


def prepare_ecare():
    """Loads the e-CARE dataset and reformats it to HuggingFace Dataset"""

    with zipfile.ZipFile("e-CARE.zip") as z:
        with z.open("dataset/train_full.jsonl") as f:
            train_df = pd.read_json(f, lines=True)
        with z.open("dataset/dev_full.jsonl") as f:
            dev_df = pd.read_json(f, lines=True)

    rel2fields = {"ask-for": "question", "hypothesis1": "choice1", "hypothesis2": "choice2", "index": "idx"}
    train_df.rename(rel2fields, axis=1, inplace=True)
    dev_df.rename(rel2fields, axis=1, inplace=True)

    train_dict = train_df.to_dict(orient="list")
    dev_dict = dev_df.to_dict(orient="list")
    ecare_train = datasets.Dataset.from_dict(train_dict)
    ecare_dev = datasets.Dataset.from_dict(dev_dict)

    return ecare_train, ecare_dev


def prepare_copa():
    """Loads the COPA dataset"""

    copa = datasets.load_dataset("super_glue", "copa")
    return copa["train"], copa["validation"]


def get_prompts_with_labels(train_set, dev_set, k_shot, explain):
    """Gets prompts together with labels"""

    prompter = Prompter(train_set, k_shot=k_shot, explain=explain)
    prompts = dev_set.map(
        prompter.make_prompt, batched=False,
        remove_columns=['premise', 'choice1', 'choice2', 'question', 'idx']
    )
    return prompts


def run_gpt3(prompts):
    """Makes calls to OpenAI API and use their GPT-3 model
    Best run in a notebook"""

    global curr_prompt_idx
    gpt_preds = []
    prompts_submitted = {k: False for k in range(prompts.num_rows)}
    print(f"Started running from example #{curr_prompt_idx}")
    while True:
        try:
            if curr_prompt_idx == prompts.num_rows:
                print("Finished.")
                break
            if not prompts_submitted[curr_prompt_idx]:
                prompt = prompts[curr_prompt_idx]["prompt"]
                pred = get_gpt3_prediction(prompt)
                gpt_preds.append(pred)
                prompts_submitted[curr_prompt_idx] = True
            curr_prompt_idx += 1
        except openai.error.RateLimitError:
            print(f"Sleeping at example #{curr_prompt_idx}.")
            time.sleep(60)
            continue
        except KeyboardInterrupt:
            print(f"Interrupted at example #{curr_prompt_idx}. Pausing.")
            break
    return gpt_preds


def save_results(gpt_preds, prompts, k_shot=0, dataset="copa", explain=False, save_dir="."):
    """Saves the GPT-3 generated texts and associated prompts to a pickle file"""

    for file_type, file_content in {"preds": gpt_preds, "prompts": prompts}.items():
        filename = f"{save_dir}/{'explain_' if explain else ''}gpt3_{dataset}_{file_type}_{k_shot}shot.bin"
        with open(filename, "wb") as f:
            pickle.dump(file_content, f)
    print("Results saved.")


def run_gpt3_copa(k_shot=0):
    train_set, dev_set = prepare_copa()
    prompts = get_prompts_with_labels(train_set, dev_set, k_shot, False)
    gpt_preds = run_gpt3(prompts)
    save_results(gpt_preds, prompts, k_shot=k_shot, dataset="copa", explain=False)


def run_gpt3_ecare(k_shot=0, explain=False):
    train_set, dev_set = prepare_ecare()
    prompts = get_prompts_with_labels(train_set, dev_set, k_shot, explain)
    gpt_preds = run_gpt3(prompts)
    save_results(gpt_preds, prompts, k_shot=k_shot, dataset="ecare", explain=explain)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--explain", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default="copa")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "copa":
        run_gpt3_copa(k_shot=args.k_shot)
    elif args.dataset == "ecare":
        run_gpt3_ecare(k_shot=args.k_shot, explain=args.explain)
    else:
        raise NotImplementedError("Dataset not implemented")


if __name__ == "__main__":
    main()

        
