from prompt2model.utils.openai_tools import ChatGPTAgent
from datasets import load_from_disk
import datasets
from pathlib import Path
import logging
import argparse
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.utils import OPENAI_ERRORS
import time
logging.basicConfig(level=logging.INFO)
from prompt2model.model_executor import ModelOutput

test_dataset_root = Path("/home/chenyan3/prompt2model_test/baseline/real_datasets/datasets")
RESULT_PATH = Path("results")
RESULT_PATH.mkdir(parents=True, exist_ok=True)

def evaluate_with_gpt(task_name):
    test_dataset = load_from_disk(test_dataset_root / f"{task_name}_gpt_model")["test"]
    chat_api = ChatGPTAgent()
    outputs = []
    evaluate_length = len(test_dataset)
    for idx in range(evaluate_length):
        api_call_counter = 0
        model_input = test_dataset[idx]["model_input"]
        input_col = test_dataset[idx]["input_col"]
        while True:
            try:
                api_call_counter += 1
                if api_call_counter >= 4:
                    logging.info(f"index: {idx}")
                    logging.info(f"input: {input_col}")
                    logging.info(f"output: None")
                    outputs.append("")
                    break
                response = chat_api.generate_one_openai_chat_completion(model_input)
                output = response.choices[0]["message"]["content"]
                outputs.append(output)
                logging.info(f"index: {idx}")
                logging.info(f"input: {input_col}")
                logging.info(f"output: {output}")
                break
            except OPENAI_ERRORS as e:
                logging.error(e)
    result_dataset = datasets.Dataset.from_dict(
        {
            "input_col": test_dataset["input_col"][:evaluate_length],
            "model_input": test_dataset["model_input"][:evaluate_length],
            "output_col": test_dataset["output_col"][:evaluate_length],
            "model_output": test_dataset["model_output"][:evaluate_length],
            "output": outputs,
        }
    )
    result_dataset.save_to_disk(RESULT_PATH / f"{task_name}_gpt_results")

    # no post-filter
    GPT_PREDICTIONS = [
    ModelOutput(
        f"{example['output']}", auxiliary_info={}
    ) for example in result_dataset
]
    evaluator = Seq2SeqEvaluator()
    metric_values = evaluator.evaluate_model(
        datasets.Dataset.from_dict(test_dataset[:evaluate_length]), "model_output", GPT_PREDICTIONS, encoder_model_name="xlm-roberta-base"
    )
    print(metric_values)
    with open(RESULT_PATH / f"{task_name}.txt", "w") as result_file:
        result_file.write(f"task_name: {task_name}\n")
        for metric_name, metric_value in metric_values.items():
            result_file.write(f"{metric_name}: {metric_value}\n")
    #  post-filter
#     filtered_GPT_PREDICTIONS = [
#     ModelOutput(
#         "N/A" if "N/A" in example["output"] else example["output"], auxiliary_info={}
#     ) for example in result_dataset
# ]
#     evaluator = Seq2SeqEvaluator()
#     metric_values = evaluator.evaluate_model(
#         datasets.Dataset.from_dict(test_dataset[:evaluate_length]), "model_output", filtered_GPT_PREDICTIONS, encoder_model_name="xlm-roberta-base"
#     )
#     print(metric_values)
#     with open(RESULT_PATH / f"{task_name}_filtered.txt", "w") as result_file:
#         result_file.write(f"task_name: {task_name}\n")
#         for metric_name, metric_value in metric_values.items():
#             result_file.write(f"{metric_name}: {metric_value}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT model.")
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name of the task."
    )
    args = parser.parse_args()

    evaluate_with_gpt(args.task_name)


def merge_gpt_outputs():
    test_dataset = load_from_disk(test_dataset_root / f"SQuAD_gpt_model")["test"]
    outputs = load_from_disk("gpt_output")["output"]
    result_dataset = datasets.Dataset.from_dict(
        {
            "input_col": test_dataset["input_col"],
            "model_input": test_dataset["model_input"],
            "output_col": test_dataset["output_col"],
            "model_output": test_dataset["model_output"],
            "output": outputs,
        }
    )
    result_dataset.save_to_disk(RESULT_PATH / f"SQuAD_gpt_results")

def fix_empty_response(task_name):
    chat_api = ChatGPTAgent()
    result_dataset = load_from_disk(RESULT_PATH / f"{task_name}_gpt_results")
    input_cols = result_dataset["input_col"]
    model_inputs = result_dataset["model_input"]
    output_cols = result_dataset["output_col"]
    model_outputs = result_dataset["model_output"]
    outputs = result_dataset["output"]
    for idx, each in enumerate(outputs):
        if each == "" or each == "None":
            print(f"None found in {idx}")
            api_call_counter = 0
            while True:
                try:
                    api_call_counter += 1
                    if api_call_counter >= 4:
                        logging.info(f"index: {idx}")
                        logging.info(f"input: {input_cols[idx]}")
                        logging.info(f"output: Failed")
                        response = ""
                        break
                    response = chat_api.generate_one_openai_chat_completion(model_inputs[idx]).choices[0]["message"]["content"]
                    outputs[idx] = response
                    logging.info(f"index: {idx}")
                    logging.info(f"input: {input_cols[idx]}")
                    logging.info(f"output: {response}")
                    break
                except OPENAI_ERRORS as e:
                    logging.error(e)
                    time.sleep(1)
    result_dataset = datasets.Dataset.from_dict(
        {
            "input_col": input_cols,
            "model_input": model_inputs,
            "output_col": output_cols,
            "model_output": model_outputs,
            "output": outputs,
        }
    )
    result_dataset.save_to_disk(RESULT_PATH / f"{task_name}_gpt_results")

def test_nq():
    result_dataset= load_from_disk(RESULT_PATH / "NQ_gpt_results")
    from datasets import load_dataset
    dataset = load_dataset("nq_open", split="validation")
    right_num = 0
    for idx, output in enumerate(result_dataset["output"]):
        print(output)
        if output in dataset[idx]["answer"]:
            right_num += 1
    print(right_num / len(result_dataset))


# if __name__ == "__main__":
#     main()