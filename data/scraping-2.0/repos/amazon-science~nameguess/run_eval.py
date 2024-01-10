import os
import sys
import json
import argparse
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], "./src"))
print(sys.path)
from metric import BNGMetrics
from model_call import OpenaiLLM


def extract_answer(raw_answer_str: str, sep_token: str):
    raw_answer_str = raw_answer_str.strip("").split(".")[0]
    answer_list = [_ans.strip("") for _ans in raw_answer.split(sep_token)]
    return answer_list


class PromptTemplate:
    @property
    def demos(self):
        _demo = (
            "As abbreviations of column names from a table, "
            "c_name | pCd | dt stand for Customer Name | Product Code | Date. "
        )
        return _demo

    @property
    def sep_token(self):
        _sep_token = " | "
        return _sep_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, choices=["gpt-3.5-turbo", "gpt-4"]
    )
    parser.add_argument("--test_file_sf", type=str, default="eval_sf.json")
    parser.add_argument("--test_file_chicago", type=str, default="eval_chicago.json")
    parser.add_argument("--test_file_la", type=str, default="eval_la.json")
    args = parser.parse_args()
    DATA_NAME = "nameGuess"

    def load_data(file_name):
        return json.load(open(os.path.join("data", file_name), "r"))

    data_sf = load_data(args.test_file_sf)
    data_chicago = load_data(args.test_file_chicago)
    data_la = load_data(args.test_file_la)

    label_dict_l = data_sf + data_chicago + data_la
    num_examples = sum([len(ele["gt_label"]) for ele in label_dict_l])
    print("num_examples:", num_examples)

    # define the model and used prompt
    defined_prompt = PromptTemplate()
    model = OpenaiLLM(model_name=args.model_name)
    all_results = []
    for _idx, example_dict in enumerate(label_dict_l):
        prompt = (
            example_dict["table_prompt"] + defined_prompt.demos + example_dict["query"]
        )
        x_list, y_list = example_dict["technical_name"], example_dict["gt_label"]
        raw_answer = model(prompt, temperature=0.0, max_tokens=1024)

        # postprocessing the answers
        answers = extract_answer(raw_answer, defined_prompt.sep_token)
        if len(answers) != len(x_list):
            y_pred_list = [" "] * len(x_list)
            print("Error! The extracted answers are not correct.")
        else:
            y_pred_list = answers
            for _x, _pred, _y in zip(x_list, y_pred_list, y_list):
                print(f"{_x}\t-->\t{_pred}\t(label={_y})")

        # save the prediction and table information for each input query name
        one_results = []
        for _x, _y, _pred, _level in zip(
            x_list, y_list, y_pred_list, example_dict["difficulty"]
        ):
            one_results.append(
                [
                    str(_idx),
                    str(example_dict["table_partition"]),
                    str(example_dict["table_id"]),
                    str(_x),
                    str(_y),
                    str(_pred),
                    str(_level),
                ]
            )
        all_results += one_results

    pred_df = pd.DataFrame(
        all_results,
        columns=[
            "example_id",
            "table_partition",
            "table_id",
            "technical_name",
            "gt_label",
            "prediction",
            "difficulty",
        ],
    )
    print("pred_df:", pred_df)

    # Evaluate the individual accuracy for each prediction
    metric_names = ["squad"]
    metric_generator = BNGMetrics(metric_names)  # , #device=device)

    individual_scores = metric_generator.compute_scores(
        predictions=pred_df["prediction"], references=pred_df["gt_label"], level="individual"
    )
    pred_df["exact-match"] = individual_scores["individual_squad-em"]
    pred_df["f1"] = individual_scores["individual_squad-f1"]

    # save the results
    save_res = {
        "squad-em": individual_scores["squad-em"],
        "squad-f1": individual_scores["squad-f1"],
        "squad-pm": individual_scores["squad-pm"],
        "model-name": args.model_name,
        "total-num-example": num_examples,
        "demo": defined_prompt.demos,
    }
    print(json.dumps(save_res, indent=4))

    # save results and configures
    save_dir = os.path.join('outputs', "{}-{}-results".format(DATA_NAME, args.model_name))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f"{args.model_name}_results.json"), "w") as fo:
        json.dump(save_res, fo, indent=4)
    # save individual prediction results
    pred_df.to_csv(os.path.join(save_dir, f"{args.model_name}_predictions.csv"))
