# Import the openai package
import openai
import logging
import random
import numpy as np


from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tenacity import retry, stop_after_attempt, wait_random_exponential # for exponential backoff of requests

from core.data_processing.flan_data_processing import get_new_train_test_split, get_best_config, get_data, \
    get_targets_and_preds
from core.data_processing.se_dataset import SelfExplanations


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def ask_gpt(user_msg, system_msg, i):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        max_tokens=10,
        temperature=0,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}
                  ])

    content = response["choices"][0]["message"]["content"]
    return content

# Set openai.api_key to the OPENAI environment variable
openai.api_key = "secret"

logging.basicConfig(level=logging.INFO)
logging.info("Starting program")

self_explanations = SelfExplanations()
logging.info("Loading SEs")
self_explanations.parse_se_from_csv("../data/results_se_aggregated_dataset_clean.csv")
logging.info("Loaded SEs")

random.seed(13)
sentence_mode = "target"

df_train, df_dev, df_test = get_new_train_test_split(self_explanations.df, sentence_mode)

system_msgs = [
    "You are a researcher grading reading comprehension strategies for students who have read STEM texts. Give short answers.",
    "You are a teacher grading reading comprehension strategies for students who have read STEM texts. Give short answers.",
    "You are a teacher grading reading comprehension strategies. Give short answers.",
    ""
]

for sys_msg_id in [0]:
    for num_examples in [0, 1, 2]:
        for num_classes, task_name, task_df_label in [
            (4, "overall", SelfExplanations.OVERALL),
            (3, "paraphrasing", SelfExplanations.PARAPHRASE),
            (3, "elaboration", SelfExplanations.ELABORATION),
            (2, "bridging", SelfExplanations.BRIDGING),
        ]:
            config = get_best_config()
            logging.info("Generating training data %d", len(df_train))
            sentences_train, targets_train = get_data(df_train, df_train, task_df_label, task_name, num_examples, config)
            logging.info("Generating dev data %d", len(df_dev))
            sentences_dev, targets_dev = get_data(df_dev, df_train, task_df_label, task_name, num_examples, config)
            logging.info("Generating test data %d", len(df_test))
            sentences_test, targets_test = get_data(df_test, df_train, task_df_label, task_name, num_examples, config)

            targets_test = targets_test
            # Define the system message
            system_msg = system_msgs[sys_msg_id]

            # Define the user message
            grades = ["A", "B", "C", "D"]

            targets = []
            predictions = []
            for i in range(len(sentences_test)):
                # use the model to generate the output
                output = ask_gpt(sentences_test[i], system_msg, i)

                targets.append(targets_test[i])
                predictions.append(output)
                if i % 100 == 0:
                    logging.info(f"------Seen {i} batches.")

            logging.info("=" * 33)
            logging.info(predictions)
            logging.info(targets)
            targets_opt, preds_opt = get_targets_and_preds(predictions, targets, grades, targets_raw_flag=False,
                                                           is_optimistical=True)
            targets_opt = np.array(targets_opt)
            preds_opt = np.array(preds_opt)
            logging.info(f"Optimistic estimation")
            logging.info(
                f"task:{task_name} details:opt-chatgpt(msg_id={sys_msg_id})-{num_examples} f1:{f1_score(targets_opt, preds_opt, average='weighted')}")
            logging.info(classification_report(targets_opt, preds_opt))
            logging.info(confusion_matrix(targets_opt, preds_opt))
            logging.info("=" * 33)
            targets_pes, preds_pes = get_targets_and_preds(predictions, targets, grades, targets_raw_flag=False,
                                                           is_optimistical=False)
            targets_pes = np.array(targets_pes)
            preds_pes = np.array(preds_pes)
            logging.info(f"Pessimistic estimation")
            logging.info(
                f"task:{task_name} details:pes-chatgpt(msg_id={sys_msg_id})-{num_examples} f1:{f1_score(targets_pes, preds_pes, average='weighted')}")
            logging.info(classification_report(targets_pes, preds_pes))
            logging.info(confusion_matrix(targets_pes, preds_pes))
            logging.info("=" * 33)
            logging.info(
                f"Sentences: {len(sentences_test)}\tOptimistic: {len(targets_opt)}\tPessimistic: {len(targets_pes)}\tPerc: {100.0 * len(targets_pes) / len(sentences_test)}")
            logging.info("=" * 33)