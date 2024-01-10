import os
import shutil
import warnings

from .instruction import SumInstruction

warnings.filterwarnings("ignore")
import pandas as pd

from .data_utils import read_text


def train_sum(epoch, instruction, example, **kwargs):
    task_name = "generative_summarization"
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
    id_tr_df = read_text(id_train_file_path, "train")
    id_te_df = read_text(id_test_file_path, "test")

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

    for batch in id_te_df.sample(100).to_dict(orient="records"):

        reference_summaries = []
        generated_summaries = []

        prompt = SumInstruction().prepare_input(batch['text'])
        response, _ = chat(prompt, **kwargs)
        generated_summaries.append(response)
        reference_summaries.append([batch['label']])

        from nltk.translate.bleu_score import sentence_bleu

        bleu_scores1 = []
        bleu_scores2 = []
        bleu_scores3 = []
        for i in range(len(generated_summaries)):
            bleu_scores1.append(sentence_bleu(reference_summaries[i], generated_summaries[i], weights=(1, 0, 0, 0)))
            bleu_scores2.append(sentence_bleu(reference_summaries[i], generated_summaries[i], weights=(0, 1, 0, 0)))
            bleu_scores3.append(sentence_bleu(reference_summaries[i], generated_summaries[i], weights=(0, 0, 1, 0)))

        metrics = {
            "rouge-1": sum(bleu_scores1) / len(bleu_scores1),
            "rouge-2": sum(bleu_scores2) / len(bleu_scores2),
            "rouge-3": sum(bleu_scores3) / len(bleu_scores3),
        }

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
