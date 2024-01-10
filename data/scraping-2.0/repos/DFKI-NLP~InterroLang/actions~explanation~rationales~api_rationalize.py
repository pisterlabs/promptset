import numpy as np
import openai
import pandas as pd
import wandb
import yaml
from pprint import pprint
from tqdm import tqdm

from actions.explanation.da_explanation import MAPPING as DA_MAPPING


MODEL_ID = "gpt-4" #"gpt-3.5-turbo"
DATASET_ID = "daily_dialog"
NUMBER_OF_INSTANCES = 200
ROOT_DIR = "../../.."
OPENAI_KEY_YAML = f"{ROOT_DIR}/configs/openai_api_key.yaml"
CACHE_DIR = f"{ROOT_DIR}/cache"
DATA_DIR = f"{ROOT_DIR}/data"


class ChatGPTHandler:
    def __init__(self, model: str = None):
        with open(OPENAI_KEY_YAML, "r") as stream:
            openai_credentials = yaml.safe_load(stream)
        openai.api_key = openai_credentials["api_key"]
        openai.organization = openai_credentials["organization"]

        self.avail_models = []
        self.responses = []
        self.wandb_true = False
        self.model = model

    def use_wandb(self) -> None:
        import wandb
        self.wandb_true = True
        wandb.init(project="InterroLang_Rationales")
        self.prediction_table = wandb.Table(columns=["prompt", "id", "completion"])

    def list_models(self) -> list:
        models = openai.Model.list()
        for i in models["data"]:
            self.avail_models.append(i["id"])
        print(self.avail_models)
        return self.avail_models

    def chat_request(self, prompt: str = None, temperature: float = 0.5, max_tokens: int = 300, top_p: float = 1.0,
                     stop: str = "\n", id=None) -> list:
        if prompt is not None:
            prompt = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        self.responses.append(response)

        if self.wandb_true:
            self.prediction_table.add_data(prompt, id, response["choices"][0]["message"]["content"])
        return response

    def visualize(self) -> None:
        if self.wandb_true:
            import wandb
            wandb.log({"predictions": self.prediction_table})
            wandb.finish()
        else:
            pprint(self.responses)


if __name__ == "__main__":
    if DATASET_ID == "boolq":
        data = pd.read_csv(f"{DATA_DIR}/boolq_validation.csv")
        explanations = pd.read_json(f"{CACHE_DIR}/boolq/ig_explainer_boolq_explanation.json")

        data["text"] = "Question: " + data["question"] + "\nPassage: " + data["passage"]

        label_mapping = {0: "false",
                         1: "true"}

        text_desc = "question and passage"
        text_fields = ["question", "passage"]
        out_desc = "answer"

        fields_enum = ', '.join([f"'{f}'" for f in text_fields])

        prevent_label_leakage_str = f"Without using {fields_enum}, " \
                                    f"or revealing the answer or outcome in your response"

    elif DATASET_ID == "daily_dialog":
        data = pd.read_csv(f"{DATA_DIR}/da_test_set_with_indices.csv")
        explanations = pd.read_json(f"{CACHE_DIR}/daily_dialog/ig_explainer_daily_dialog_explanation.json")

        data["text"] = "Text: '" + data["dialog"] + "'"

        label_mapping = DA_MAPPING
        text_desc = "the text"
        out_desc = "predicted dialogue act"
        prevent_label_leakage_str = f"Without revealing the predicted dialogue act label in your response"

    elif DATASET_ID == "olid":
        data = pd.read_csv(f"{DATA_DIR}/offensive_val.csv")
        explanations = pd.read_json(f"{CACHE_DIR}/olid/ig_explainer_olid_explanation.json")

        data["text"] = "Tweet: '" + data["text"] + "'"

        label_mapping = {0: "non-offensive",
                         1: "offensive"}

        text_desc = "the tweet"
        out_desc = "predicted label"
        prevent_label_leakage_str = f"Without revealing the predicted label in your response"
    else:
        raise ValueError(f"Dataset {DATASET_ID} not a valid choice.")

    data["prediction"] = [np.argmax(x) if type(x) != str else x for x in explanations["predictions"]]

    # Reduce amount of data
    data = data[:NUMBER_OF_INSTANCES]

    handler = ChatGPTHandler(model=MODEL_ID)
    handler.use_wandb()
    for i, (index, instance) in enumerate(tqdm(data.iterrows(), total=len(data))):
        try:
            idx = instance["idx"]
            text = instance["text"]
            prediction = instance["prediction"]
        except KeyError:
            raise "Dataset does not contain the required idx, text, and prediction columns."

        if type(prediction) == int:
            # BoolQ, OLID
            pred_str = label_mapping[prediction]
        elif DATASET_ID == "daily_dialog":
            other_labels = [v for v in list(label_mapping.values())[1:]]
            pred_str = prediction + f" (out of {', '.join(other_labels)})"
        else:
            raise TypeError(f"Invalid type for {prediction}")

        instruction_str = f"Based on {text_desc}, the {out_desc} is {pred_str}. " \
            f"{prevent_label_leakage_str}, explain why: "

        prompt = f"{text}\n{instruction_str}"
        handler.chat_request(prompt, id=idx)

        if i % 5 == 0:
            wandb.log({"predictions": handler.prediction_table})

    handler.visualize()
