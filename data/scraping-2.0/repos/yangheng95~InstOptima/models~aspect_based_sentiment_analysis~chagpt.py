import os
import shutil
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

from .data_utils import read_json


def train_absa(epoch, instruction, example, **kwargs):
    task_name = "aspect_based_sentiment_analysis"
    experiment_name = kwargs.get("dataset")
    model_checkpoint = kwargs.get("plm")
    # model_checkpoint = "google/flan-t5-small"

    print("Experiment Name: ", experiment_name)
    model_out_path = "checkpoints"
    model_out_path = os.path.join(
        model_out_path,
        task_name,
        f"{model_checkpoint.replace('/', '')}-{experiment_name}",
    )
    print("Model output path: ", model_out_path)

    id_train_file_path = kwargs.get("dataset")
    id_test_file_path = kwargs.get("dataset")
    id_tr_df = read_json(id_train_file_path, "train")
    id_te_df = read_json(id_test_file_path, "test")

    id_tr_df = pd.DataFrame(id_tr_df)
    id_te_df = pd.DataFrame(id_te_df)

    def chat(prompt, **kwargs):
        params = {"prompt": None, "message_history": None, "password": "Password!"}
        chatter_url = "https://chatter.pagekite.me/chat"
        import requests
        import time
        params.update(
            {
                "prompt": prompt,
                "system_prompt": "You are ChatGPT, a chatbot that uses the GPT-3 language model from OpenAI to answer questions about the world.",
                "message_history": '[]',
                "tag": "EvoPrompt-" + kwargs.get("dataset", "absa-chagpt-ablation-no-instructions"),
            }
        )
        try:
            response = requests.post(chatter_url, params=params, timeout=600)

            if "error" in response.json():
                print(response.json())
                time.sleep(5)
                return chat(prompt)
            if response.status_code != 200:
                print(response.status_code)
                print(response.text)
                time.sleep(5)
                return chat(prompt)
            return response.json()["response"], response.json()["message_history"]
        except Exception as e:
            print(e)
            time.sleep(5)
            return chat(prompt)

    num_total = 0
    num_acc = 0
    for batch in id_te_df.sample(100).to_dict(orient="records"):
        print(batch)
        print(batch['text'])
        print(batch['labels'])
        for label in batch['labels']:
            if label['aspect'] != 'NULL':
                prompt = f"{instruction}\n Examples: {example}\n what is the sentiment for {label['aspect']} in the following text: {batch['text']}?\n\n"
                response, _ = chat(prompt, **kwargs)
                print(response)
                if label['polarity'].lower() in response.lower():
                    num_acc += 1
                num_total += 1
                print(num_acc / num_total)

    metrics = {'accuracy': num_acc / num_total}
    # # Compute Metrics
    # metrics = t5_exp.get_metrics(id_tr_labels, id_tr_pred_labels)
    # print('----------------------- Training Set Metrics -----------------------')
    # print(metrics)
    #
    # metrics = t5_exp.get_metrics(id_te_labels, id_te_pred_labels)
    # print('----------------------- Testing Set Metrics -----------------------')
    # print(metrics)

    # # Compute Metrics
    # metrics = t5_exp.get_classic_metrics(id_tr_labels, id_tr_pred_labels)

    # print("----------------------- Classic Training Set Metrics -----------------------")
    # print(metrics)

    print("----------------------- Classic Testing Set Metrics -----------------------")
    print(metrics)

    try:
        shutil.rmtree("checkpoints")
    except:
        pass
    return metrics
