import argparse
import os
from pathlib import Path
from typing import List, Dict

from gptinference import utils
from gptinference.base_prompt import Prompt
from gptinference.openai_wrapper import OpenAIWrapper


class MeanResponseQA(Prompt):
    def __init__(self, engine: str, openai_wrapper: OpenAIWrapper):
        super().__init__()
        self.openai_wrapper = openai_wrapper
        self.engine = engine

    def make_query(self, experiment_type: str, ques: str, choices: List[str]) -> str:
        if not ques:
            return ""

        # Thinking of yourself as a person, please select the right choice.
        #
        # Thinking about your spouse or partner is your spouse or partner of the opposite sex as you or the same sex as you?
        #
        # Choice: [ "Opposite sex",  "Same sex", "Refused"]
        choices_str = ", ".join([f'"{c}"' for c in choices])
        #         "most_freq_agreed_answers": [
        #             [
        #                 0.5336318932209343,
        #                 "Some"
        #             ],
        #             [
        #                 0.2794357469015003,
        #                 "Only a few"
        #             ]
        #         ],
        #         "polparty_info": {
        #             "Independent": [
        #                 [
        #                     0.4061433447098976,
        #                     "Some"
        #                 ],
        #                 [
        #                     0.2593856655290102,
        #                     "Only a few"
        #                 ]
        #             ],

        role = "person"
        if experiment_type.startswith("ideology"):
            role = experiment_type.split(":")[-1]

        prompt = \
f"""Thinking of yourself as a {role.lower()}, please select the right choice.

{ques.strip()}

Choice: [{choices_str.strip()}]\n
"""
        return prompt

    def __call__(self, experiment_type: str, ques: str, choices: List[str]) -> str:
        generation_query = self.make_query(experiment_type=experiment_type, ques=ques, choices=choices)

        generated_sent = self.openai_wrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=10,
            stop_token="###",
            temperature=0.0
        )

        return generated_sent.replace("Answer:", "").strip()  # gpt3 turbo adds newline in the beginning so strip it.


def compare_ans(reference: str, gpt_output: str):
    ref = utils.only_a_z(reference)
    pred = utils.only_a_z(gpt_output)
    exact_match = ref == pred
    starts_with = ref.startswith(pred)  # when max. tokens are not enough.
    # Rather than increase costs with more max_tokens, check if answer startswith ref.

    return exact_match or starts_with


def get_gold(ques_j, experiment_type):
    try:
        gold = ques_j["most_freq_agreed_answers"]
        if experiment_type.startswith("ideology"):
            gold = ques_j["polparty_info"][experiment_type.split(":")[-1]]
        return gold[0][1]  # Extract "Some" from [ [0.53, "Some"], [0.27, "Only a few"] ]
    except Exception as exc:
        print(f"Could not extract ({exc})\njson:\n{ques_j}\n")
        return "ERROR"

def ensure_path(fp: str, is_dir: bool):
    # Create parent level subdirectories if not exists.
    p = Path(fp)
    if os.path.exists(fp):
        return
    # if path indicates a dir: Create parent level subdirectories if not exists.
    if not is_dir and not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    # if path indicates a file: Create parent level subdirectories if not exists
    elif not p.exists() and is_dir:
        p.mkdir(parents=True, exist_ok=True)


def collapsed_class(choices, choice):
    total_len = len(choices) - 1
    # if total_len % 2 == 0: # even number -> mid=2
    mid = total_len // 2
    if choice in choices[:mid + (1 if total_len%2 ==1 else 0)]:  # incl. the mid.
        return "less-extreme-bucket"
    else:
        return "more-extreme-bucket"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # "data/qa-pairs/random_question_list.json", data/qa-pairs/random_question_list.json
    parser.add_argument("--in_path", type=str, default="data/qa-pairs/random_question_list.json", help="json path")
    parser.add_argument("--out_dir", type=str, default="analysis/mean_responses/", help="json path")
    parser.add_argument("--cache_path", type=str, default="data/cache/gpt_cache.jsonl", help="json path")
    parser.add_argument("--max_retries", type=int, default=2, help="max number of openai retries when a call fails.")
    parser.add_argument("--experiment_type", type=str, required=True, help="ideology:Independent, ideology:Republican, ideology:Democrat, overall")
    parser.add_argument("--engine", type=str, default="text-davinci-003",
                        help="which model to use. (choices: text-davinci-003, gpt-3.5-turbo)")

    args = parser.parse_args()
    os.environ["OPENAI_MAX_TRIES_INT"] = str(args.max_retries)
    out_path = os.path.join(args.out_dir, f"{args.experiment_type}/model_output.jsonl")  # "data/mean_responses_100q/model_output.jsonl"
    eval_out_path = os.path.join(args.out_dir, f"{args.experiment_type}/model_accuracy.jsonl")  # "data/mean_responses_100q/model_output.jsonl"

    # "ideology:Independent", "ideology:Republican", "ideology:Democrat", "overall:most_freq_agreed_answers"
    # same setup with answers "per" demography e.g., what do democrats say and then change role.

    outputs = []
    evals = []
    evals_collapsed = []

    gpt_inference_obj = MeanResponseQA(engine=args.engine, openai_wrapper=OpenAIWrapper(cache_path=args.cache_path))
    for ques_j in utils.read_jsonl_or_json(args.in_path):
        try:
            gpt_output = gpt_inference_obj(experiment_type=args.experiment_type, ques=ques_j["question"], choices=ques_j["choice"])
        except Exception as exc:
            print(f"Exception in mean response qa: {exc}")
            gpt_output = ""
        gold = get_gold(ques_j, args.experiment_type)
        is_correct = compare_ans(reference=gold, gpt_output=gpt_output)
        is_collapsed_correct = compare_ans(reference=collapsed_class(ques_j["choice"], gold), gpt_output=collapsed_class(ques_j["choice"], gpt_output))
        outputs.append({
            "type": args.experiment_type,
            "is_correct": is_correct,
            "is_collapsed_correct": is_collapsed_correct,
            "gold": gold,
            "predicted": gpt_output,
            "ques": ques_j
        })
        evals.append(int(is_correct))
        evals_collapsed.append(int(is_collapsed_correct))

    overall_accuracy = sum(evals)/ len(evals)
    collapsed_overall_accuracy = sum(evals_collapsed)/ len(evals_collapsed)
    p = f"\nAverage accuracy of {args.engine} on {len(evals)} questions in the {args.experiment_type} experiment:\n" \
        f"Exact match: {overall_accuracy: 0.3f}\n" \
        f"Collapsed  : {collapsed_overall_accuracy: 0.3f}"
    print(p)

    ensure_path(out_path, is_dir=False)
    utils.write_jsonl(outpath=out_path, data_points=outputs)

    ensure_path(eval_out_path, is_dir=False)
    utils.write_jsonl(data_points=[p] + [f"{x: 0.3f}".strip() for x in evals], outpath=eval_out_path)
    print(f"\nSee {eval_out_path}\n{out_path}")



