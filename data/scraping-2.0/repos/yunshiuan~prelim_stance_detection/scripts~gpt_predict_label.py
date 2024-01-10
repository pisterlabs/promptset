"""
Label the stances of the tweets with LMM like GPT.
@author: Yun-Shiuan Chuang
@contact: yunshiuan.chuang@gmail.com
"""
import os
from os.path import join
import pandas as pd
import numpy as np
import datetime
import re
import torch as th
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from accelerate import Accelerator

from transformers import Trainer
from collections import defaultdict

from data_processor import SemEvalDataProcessor
from gpt_data_processor import SemEvalGPTDataProcessor
from utils import preprocess_dataset, flatten_1d_nested_list, creat_dir_for_a_file_if_not_exists, save_dict_as_yaml, get_dir_of_a_file, wait_for_user_input, partition_and_resample_df, get_parameters_for_dataset, logits_to_log_probs, index_logits_by_1d_index_tensor, AttrDict
import openai
import tiktoken
from time import sleep

# - https://platform.openai.com/tokenizer
TOKENS_PER_WORD_ESTIMATE = 4 / 3


def main(config):
    """Label the stances of the tweets with LMM like GPT.

    Args:
        config (argparse.Namespace): the configuration of the experiment.
    """
    # ----------
    # save the config of this run to a yaml file
    # ----------
    # convert the config into an AttrDict to access keys as attributes
    config = AttrDict(config)

    assert config.partition_type in ["single_domain", "gda", "all"], "The partition type must be one of ['single_domain', 'gda', 'all']."
    # - save to the output directory
    path_output_predictions = get_dir_of_a_file(config.file_output_predictions)
    creat_dir_for_a_file_if_not_exists(config.file_output_predictions)
    save_dict_as_yaml(join(path_output_predictions, "config_{}_{}_{}_predict.yaml".format(config.version_output, config.partition_type, config.topic)), config)

    # --------------------
    # Load data
    # --------------------
    if config.dataset_meta == "SEM_EVAL":
        gpt_data_processor = SemEvalGPTDataProcessor(config.version_prompt, topic=config.topic)
        df_input_text = gpt_data_processor.embed_prompt(file_output=None, write_csv=False,
                                                        return_df=True)
        sem_eval_data = SemEvalDataProcessor()
        df_partitions = sem_eval_data.read_partitions(topic=config.topic)
    else:
        raise NotImplementedError("The dataset_meta must be 'COVID_VACCINE'.")

    # --------------------
    # Partition the data
    # - useful to save cost when using OpenAI's API
    # - e.g., for single_domain, only predict the labels for vali and test (but not train)
    # - e.g., for gda, only predict the labels for target domains and source valia and source test (but not source train)
    # --------------------
    if config.partition_type != "all":

        # partition the data
        if config.partition_type == "single_domain":
            # - single domain split
            dict_df_single_domain = partition_and_resample_df(
                df_input_text, seed=None, partition_type="single_domain",
                read_partition_from_df=True,
                df_partitions=df_partitions)
        elif config.partition_type == "gda":
            # - GDA split
            dict_df_gda = partition_and_resample_df(
                df_input_text, seed=None, partition_type="gda",
                list_time_domain=config.list_time_domain,
                list_source_domain=config.list_source_domain,
                partition_source_method=["train", "vali", "test"],
                partition_target_method=["all", "vali", "test"],
                read_partition_from_df=True,
                df_partitions=df_partitions)

        # select the partition to be labeled
        # - single_domain: vali and test
        # - gda: target domains and source vali and source test
        df_input_text_filtered = pd.DataFrame()
        if config.partition_type == "single_domain":
            for partition in ["vali_raw", "test_raw"]:
                df_input_text_filtered = pd.concat([df_input_text_filtered, dict_df_single_domain[partition]])
        elif config.partition_type == "gda":
            for partition in ["source_vali_raw", "source_test_raw"]:
                df_input_text_filtered = pd.concat([df_input_text_filtered, dict_df_gda[partition]])
            # use regex to match all "target_*_all" partitions in dict_df_gda
            for partition in dict_df_gda.keys():
                if re.match(r"target_.*_all", partition):
                    # print("partition: {}".format(partition))
                    df_input_text_filtered = pd.concat([df_input_text_filtered, dict_df_gda[partition]])

        df_input_text = df_input_text_filtered
        # - limit the number of texts to be labeled (for debugging)
        # -- This is useful for debugging while keeping the cost minimal (money cost if OpenAI's API is used).
        if config.limit_first_n is not None:
            df_input_text = df_input_text[:config.limit_first_n]
    # --------------------
    # Make predictions
    # --------------------
    list_label_space = None
    if config.model_type == "flan-t5-xxl":
        llm_label_predictor = FlanT5XxlLabelPredictor(col_name_text="tweet_embedded",
                                                      col_name_label="stance_predicted",
                                                      col_name_text_id=par.TEXT_ID,
                                                      try_use_gpu=config.try_use_gpu,
                                                      per_device_eval_batch_size=config.per_device_eval_batch_size,
                                                      mode_output=config.mode_output,
                                                      for_generation_only=config.for_generation_only
                                                      )
        if config.dataset_meta == "COVID_VACCINE":
            list_label_space = ["in-favor", "against", "neutral-or-unclear"]
        # [16, 18, 89, 9, 1967, 1]
        # [581, 1]
        # [7163, 18, 127, 18, 202, 2482, 291, 1]
    elif config.model_type == "flan-t5-large":
        llm_label_predictor = FlanT5LargeLabelPredictor(col_name_text="tweet_embedded",
                                                        col_name_label="stance_predicted",
                                                        col_name_text_id=par.TEXT_ID,
                                                        try_use_gpu=config.try_use_gpu,
                                                        per_device_eval_batch_size=config.per_device_eval_batch_size,
                                                        mode_output=config.mode_output,
                                                        for_generation_only=config.for_generation_only
                                                        )
        if config.dataset_meta == "COVID_VACCINE":
            list_label_space = ["in-favor", "against", "neutral-or-unclear"]
        # [16, 18, 89, 9, 1967, 1]?
        # [581, 1]?
        # [7163, 18, 127, 18, 202, 2482, 291, 1]?
    elif config.model_type == "flan-t5-small":
        llm_label_predictor = FlanT5SmallLabelPredictor(col_name_text="tweet_embedded",
                                                        col_name_label="stance_predicted",
                                                        col_name_text_id=par.TEXT_ID,
                                                        try_use_gpu=config.try_use_gpu,
                                                        per_device_eval_batch_size=config.per_device_eval_batch_size,
                                                        mode_output=config.mode_output,
                                                        for_generation_only=config.for_generation_only
                                                        )
        if config.dataset_meta == "COVID_VACCINE":
            list_label_space = ["in-favor", "against", "neutral-or-unclear"]
        # [16, 18, 89, 9, 1967, 1]?
        # [581, 1]?
        # [7163, 18, 127, 18, 202, 2482, 291, 1]?
    elif config.model_type == "flan_ul2":
        llm_label_predictor = FlanUl2LabelPredictor(col_name_text="tweet_embedded",
                                                    col_name_label="stance_predicted",
                                                    col_name_text_id=par.TEXT_ID,
                                                    try_use_gpu=config.try_use_gpu,
                                                    per_device_eval_batch_size=config.per_device_eval_batch_size,
                                                    mode_output=config.mode_output
                                                    )
        if config.dataset_meta == "COVID_VACCINE":
            list_label_space = ["in-favor", "against", "neutral"]
    elif config.model_type == "chatgpt_turbo_3_5":
        llm_label_predictor = GPTChatTurbo3_5LabelPredictor(col_name_text="tweet_embedded",
                                                            col_name_label="stance_predicted",
                                                            col_name_text_id=par.TEXT_ID,
                                                            mode_output=config.mode_output)
        total_cost_estimate = llm_label_predictor.estimate_total_cost(df_input_text)
        print("Estimated total cost: {}".format(total_cost_estimate))
        if config.dataset_meta == "COVID_VACCINE":
            list_label_space = ["In Favor", "Against", "Neutral"]
        # check if the author indeed wants to proceed with the prediction (which costs money)
        # - wait for y/n input
        if not wait_for_user_input():
            # Terminate the program
            return
    else:
        raise ValueError(f"model_type {config.model_type} is not supported.")

    llm_label_predictor.predict_labels(df_input_text,
                                       config.file_output_predictions,
                                       keep_tweet_id=True, keep_text=False,
                                       output_prob_mode=config.output_prob_mode,
                                       list_label_space=list_label_space,
                                       col_name_tweet_id=par.TEXT_ID)


class LMMLabelPredictor():
    def __init__(self, col_name_text, col_name_label, col_name_text_id, name_model) -> None:
        """
        Args:
            col_name_text (str): the name of the column that contains the texts to be labeled.
            col_name_label (str): the name of the column that contains the predicted labels.
            col_name_text_id (str): the name of the column that contains the text id.
            name_model(str): the name of the model to be used for labeling.
        """
        self.col_name_text = col_name_text
        self.col_name_label = col_name_label
        self.col_name_text_id = col_name_text_id
        self.name_model = name_model

    def predict_labels(self, df_input_text, file_output_predictions):
        """Label the stances of the texts with LMM like GPT.

        Args:
            df_input_text (pd.DataFrame): the df that contains the texts to be labeled.
            file_output_predictions (str): the path of the output predictions.
        """
        raise NotImplementedError()

    def _load_model(self):
        """Load the model for labeling."""
        raise NotImplementedError()


class GPTLabelPredictor(LMMLabelPredictor):
    """Label the stances of the texts with GPT-based models. This involves interacting with the model with OpenAI's API.
    - inspired by this tutorial:
        - https://colab.research.google.com/drive/1M986FV4IGx6zQ3kRMdDpdIOIa6kzJ1mE?authuser=2#scrollTo=g6RibKGN6Rum
    - documentation:
        - openai.ChatCompletion.create()
            - https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
    """

    def __init__(self, col_name_text, col_name_label, col_name_text_id, name_model) -> None:
        """
        Args:
            col_name_text (str): the name of the column that contains the texts to be labeled.
            col_name_label (str): the name of the column that contains the predicted labels.
            col_name_text_id (str): the name of the column that contains the text id.
            name_model(str): the name of the model to be used for labeling.
        Notes:
            API key is stored in the environment variable OPENAI_API_KEY.
            https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
        """
        assert name_model in ["gpt-chat-turbo-3_5"]
        super().__init__(col_name_text, col_name_label, col_name_text_id, name_model)
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def predict_labels(self, df_input_text, file_output_predictions):
        raise NotImplementedError()

    def estimate_total_cost(self):
        raise NotImplementedError()

    def estimate_cost_per_request(self):
        """Estimate the cost per request.
        """
        raise NotImplementedError()

    def _esimate_token_per_request(self):
        """Estimate the number of tokens per request. This involves interacting with the model with OpenAI's API.
        """
        # # num_tokens_input = self._count_words_and_punctuations_in_str(str_input) * TOKENS_PER_WORD_ESTIMATE
        # num_tokens_input = len(str_input) * TOKENS_PER_WORD_ESTIMATE
        # num_tokens_output = len_str_output_estimate * TOKENS_PER_WORD_ESTIMATE
        # return num_tokens_input + num_tokens_output
        raise NotImplementedError()

    def _create_message_object(self, text):
        """Create the message object to hold the prompt."""
        raise NotImplementedError()

    # def _count_words_and_punctuations_in_str(self, str_input):
    #     """Count the number of words and punctuations in a string.

    #     Args:
    #         str_input (str): the input string.
    #     """
    #     # - https://www.geeksforgeeks.org/python-program-to-count-words-in-a-sentence/
    #     return len(re.findall(r'\w+|[^\w\s]', str_input))


class GPTInstructLabelPredictor(GPTLabelPredictor):
    """Label the stances of the texts with instruct-GPT-based models. This involves interacting with the model with OpenAI's API. Examples include "text-davinci-003" etc.
    """

    def __init__(self, col_name_text, col_name_label, col_name_text_id, name_model) -> None:
        """
        Args:
            col_name_text (str): the name of the column that contains the texts to be labeled.
            col_name_label (str): the name of the column that contains the predicted labels.
            col_name_text_id (str): the name of the column that contains the text id.
            name_model(str): the name of the model to be used for labeling.
        """
        super().__init__(col_name_text, col_name_label, col_name_text_id, name_model)


class GPTChatLabelPredictor(GPTLabelPredictor):
    """Label the stances of the texts with ChatGPT-based models. This involves interacting with the model with OpenAI's API. Examples include "gpt-3.5-turbo" etc.
    """

    def __init__(self, col_name_text, col_name_label, col_name_text_id, name_model, mode_output) -> None:
        """
        Args:
            col_name_text (str): the name of the column that contains the texts to be labeled.
            col_name_label (str): the name of the column that contains the predicted labels.
            col_name_text_id (str): the name of the column that contains the text id.
            name_model (str): the name of the model to be used for labeling.
            mode_output (str): the mode of the output, e.g., "single-word".
        - see:
            - https://platform.openai.com/docs/guides/chat/introduction
            - https://platform.openai.com/docs/api-reference/introduction
        """
        assert name_model == "gpt-chat-turbo-3_5"
        super().__init__(col_name_text, col_name_label, col_name_text_id, name_model)

        assert mode_output in ["single-word", "CoT"]
        self.mode_output = mode_output


class GPTChatTurbo3_5LabelPredictor(GPTChatLabelPredictor):
    """Label the stances of the texts with "gpt-3.5-turbo".
    """

    def __init__(self, col_name_text, col_name_label, col_name_text_id, mode_output) -> None:
        """
        Args:
            col_name_text (str): the name of the column that contains the texts to be labeled.
            col_name_label (str): the name of the column that contains the predicted labels.
            col_name_text_id (str): the name of the column that contains the text id.
            mode_output (str): the mode of the output, e.g., "single-word".
        - see:
            - https://platform.openai.com/docs/guides/chat/introduction
            - https://platform.openai.com/docs/api-reference/introduction
        """
        self.model_full_name = "gpt-3.5-turbo"
        self.cost_rate_per_token = 0.002 / 1000
        super().__init__(col_name_text, col_name_label, col_name_text_id, "gpt-chat-turbo-3_5", mode_output)
        if mode_output == "single-word":
            # note: "neutral-or-unclear" has 6 tokens
            self.MAX_OUTPUT_TOKENS = 10
        elif mode_output == "CoT":
            self.MAX_OUTPUT_TOKENS = 150

    def predict_labels(self, df_input_text, file_output_predictions, keep_text=True, keep_tweet_id=False, output_prob_mode=None, list_label_space=None, col_name_tweet_id="tweet_id"):
        """Label the stances of the texts with GPT-3.5-turbo-based models. This involves interacting with the model with OpenAI's API.

        Args:
            df_input_text (pd.DataFrame): the df that contains the texts to be labeled.
            file_output_predictions (str): the path of the output predictions.
            keep_text (bool, optional): whether to keep the text in the output predictions. Defaults to True. If false, only save the top-10 rows of the input texts.
            keep_tweet_id (bool, optional): whether to keep the tweet id in the output predictions. Defaults to False.
            output_prob_mode (str, optional): whether to output the normalized probabilities of the labels. The probability is normalzed by the `list_label_space`. Defaults to None.
            list_label_space (list, optional): the list of possible labels. Used to normalize the probabilities. Have to be provided if `output_prob_mode` is not None. Defaults to None.
            col_name_tweet_id (str): the name of the column that contains the tweet_id. Default: "tweet_id".
        """
        assert output_prob_mode is None, "output_prob_mode is not supported for GPT-3.5-turbo-based models."
        assert list_label_space is None, "list_label_space is not supported for GPT-3.5-turbo-based models."
        # --------------------
        # read existing predictions
        # - avoid re-predicting the same texts
        # --------------------
        count_text_existing = 0
        # df_input_text = df_input_text.head(30)
        if os.path.exists(file_output_predictions):
            df_predictions_existing = pd.read_csv(file_output_predictions, dtype={self.col_name_text_id: str})
            # - remove the texts that have already been predicted
            mask_duplicates = df_input_text[self.col_name_text_id].isin(df_predictions_existing[self.col_name_text_id].values)
            df_input_text = df_input_text[~df_input_text[self.col_name_text_id].isin(df_predictions_existing[self.col_name_text_id].values)]
            print("existed predictions: {}".format(len(df_predictions_existing)))
            print("new predictions to make: {}".format(len(df_input_text)))
            # count non-empty string in the text in existing predictions
            count_text_existing = sum(~df_predictions_existing[self.col_name_text].isnull())
        # --------------------
        # predict the labels
        # --------------------
        # - collect the inputs and results
        list_tweet_embedded = []
        if keep_tweet_id:
            list_tweet_id = []
        list_output_labels = []
        # - for warning messages only
        list_tweet_warnings = []
        list_message_warnings = []
        try:
            for i_tweet, (tweet_embedded, tweet_id) in enumerate(
                zip(df_input_text[self.col_name_text].values,
                    df_input_text[self.col_name_text_id].values)):
                print("tweet: {}/{}".format(i_tweet, len(df_input_text)))
                try:
                    chat = openai.ChatCompletion.create(
                        model=self.model_full_name,
                        messages=self._create_message_object(tweet_embedded),
                        temperature=0,
                        max_tokens=self.MAX_OUTPUT_TOKENS
                    )
                except Exception as e:
                    message_warning = "Tweet ID: {}. Error: {} ".format(tweet_id, str(e))
                    print(message_warning)
                    list_tweet_warnings.append(tweet_id)
                    list_message_warnings.append(message_warning)
                    sleep(30)
                    chat = openai.ChatCompletion.create(
                        model=self.model_full_name,
                        messages=self._create_message_object(tweet_embedded),
                        temperature=0,
                        max_tokens=self.MAX_OUTPUT_TOKENS
                    )
                # --------------------
                # Parse the output and collect the results
                # --------------------
                output_label = chat.choices[0].message.content
                if chat.choices[0].finish_reason != "stop":
                    message_warning = "Tweet ID: {}. Warning: the prediction is not finished. ".format(tweet_id)
                    print(message_warning)
                    list_tweet_warnings.append(tweet_id)
                    list_message_warnings.append(message_warning)
                list_tweet_embedded.append(tweet_embedded)
                if keep_tweet_id:
                    list_tweet_id.append(tweet_id)
                list_output_labels.append(output_label)
        except Exception as e:
            print("Not all tweets are predicted. Error: {}".format(str(e)))
            print("Save the predictions so far.")
        # --------------------
        # create a df for the inputs and the predictions
        # --------------------
        df_predictions_new = pd.DataFrame({self.col_name_text: list_tweet_embedded,
                                           self.col_name_label: list_output_labels})
        if keep_tweet_id:
            df_predictions_new[self.col_name_text_id] = list_tweet_id
        if keep_text:
            df_predictions_new[self.col_name_text] = list_tweet_embedded
        else:
            # only keep the top-10 rows of texts
            # - consider the texts in the existing predictions
            text_to_add = max(10 - count_text_existing, 0)
            df_predictions_new[self.col_name_text] = list_tweet_embedded[:text_to_add] + [""] * (len(list_tweet_embedded) - text_to_add)
        if os.path.exists(file_output_predictions):
            df_predictions = pd.concat([df_predictions_existing, df_predictions_new], axis=0)
        else:
            df_predictions = df_predictions_new

        # --------------------
        # write to csv
        # --------------------
        # - redorder the columns to stance_predicted,tweet_id,tweet_embedded
        df_predictions = df_predictions[["stance_predicted", self.col_name_text_id, "tweet_embedded"]]
        creat_dir_for_a_file_if_not_exists(file_output_predictions)
        df_predictions.to_csv(file_output_predictions, index=False)

        # --------------------
        # Create a df for the warnings
        # --------------------
        df_warnings = pd.DataFrame({self.col_name_text_id: list_tweet_warnings,
                                    "message": list_message_warnings})
        if len(df_warnings) > 0:
            file_output_warnings = file_output_predictions.replace(".csv", "_warnings.csv")
            df_warnings.to_csv(file_output_warnings, index=False)

    def _create_message_object(self, text):
        """Create the message object to hold the prompt."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]

    def _esimate_token_per_request(self, messages_input, num_tokens_output_estimate):
        """Estimate the number of tokens per request. This involves interacting with the model with OpenAI's API.

        Args:
            messages_input (list): the input text.
            num_tokens_output_estimate (int): the estimated number of tokens in the output text.
        - see:
            - https://platform.openai.com/tokenizer
            - https://platform.openai.com/docs/guides/chat/introduction
            - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        # https://platform.openai.com/docs/guides/chat/introduction
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        encoding = tiktoken.encoding_for_model(self.model_full_name)
        input_num_tokens = 0
        for message in messages_input:
            input_num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                input_num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    input_num_tokens += -1  # role is always required and always 1 token
        input_num_tokens += 2  # every reply is primed with <im_start>assistant
        return input_num_tokens + num_tokens_output_estimate

    def estimate_cost_per_request(self, messages_input, num_tokens_output_estimate):
        """Estimate the cost per request.

        Args:
            messages_input (list): the input text.
            num_tokens_output_estimate (int): the estimated number of tokens in the output text.
        """
        num_tokens_estimate = self._esimate_token_per_request(messages_input, num_tokens_output_estimate)
        return self.cost_rate_per_token * num_tokens_estimate

    def estimate_total_cost(self, df_input_text):
        """Estimate the total cost.
        Args:
            df_input_text (pd.DataFrame): the df that contains the texts to be labeled.
        """
        cost_total = 0
        for i_tweet, (tweet_embedded) in enumerate(df_input_text[self.col_name_text].values):
            # print("tweet: {}/{}".format(i_tweet, len(df_input_text)))
            cost_total += self.estimate_cost_per_request(self._create_message_object(tweet_embedded), self.MAX_OUTPUT_TOKENS)
        # print("Estimated total cost: {}".format(cost_total))
        return cost_total


class FlanLabelPredictor(LMMLabelPredictor):
    """Label the stances of the texts with Flan-based models
    """

    def __init__(self, col_name_text, col_name_label, col_name_text_id, name_model, try_use_gpu, per_device_eval_batch_size=16, mode_output="single-word") -> None:
        """
        Args:
            col_name_text (str): the name of the column that contains the texts to be labeled.
            col_name_label (str): the name of the column that contains the predicted labels.
            col_name_text_id (str): the name of the column that contains the text id.
            name_model(str): the name of the model to be used for labeling.
            try_use_gpu (bool): whether to use GPU if available.
            per_device_eval_batch_size (int): the batch size for each GPU. Default: 16.
            mode_output (str): the mode of the output, e.g., "single-word". Default: "single-word".
        Notes:
            - https://huggingface.co/google/flan-t5-xxl
        """
        assert name_model in ["flan-t5-xxl", "flan-ul2", "flan-t5-small", "flan-t5-large"]
        assert mode_output in ["single-word", "CoT"]
        super().__init__(col_name_text, col_name_label, col_name_text_id, name_model)

        self.try_use_gpu = try_use_gpu
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.mode_output = mode_output
        # self.accelerator = Accelerator()

        # detect the device
        if try_use_gpu:
            device = th.device('cuda' if th.cuda.is_available() else 'cpu')
            # device = self.accelerator.device
        else:
            device = th.device('cpu')
        self.device = device

    def predict_labels(self, df_input_text, file_output_predictions, keep_text=True, keep_tweet_id=False, output_prob_mode=None, list_label_space=None, col_name_tweet_id="tweet_id"):
        """Label the stances of the text.

        Args:
            df_input_text (pd.DataFrame): the df that contains the texts to be labeled.
            file_output_predictions (str): the path of the output predictions.
            keep_text (bool, optional): whether to keep the text in the output predictions. Defaults to True. If false, only save the top-10 rows of the input texts.
            keep_tweet_id (bool, optional): whether to keep the tweet id in the output predictions. Defaults to False.
            output_prob_mode (str, optional): whether to output the normalized probabilities of the labels. The probability is normalzed by the `list_label_space`. Defaults to None.
            list_label_space (list, optional): the list of possible labels. Used to normalize the probabilities. Have to be provided if `output_prob_mode` is not None. Defaults to None.
            col_name_tweet_id (str): the name of the column that contains the tweet_id. Default: "tweet_id".
        """
        # --------------------
        # load the model
        # --------------------
        self._load_model()

        # --------------------
        # convert the df to the dataset
        # --------------------
        pred_dataset = preprocess_dataset(df_input_text, self.tokenizer,
                                          col_name_text=self.col_name_text,
                                          with_label=False,
                                          keep_tweet_id=True,
                                          name_model=self.name_model,
                                          output_batched=True,
                                          output_batch_size=self.per_device_eval_batch_size,
                                          pad_to_max_in_dataset=True,
                                          col_name_tweet_id=col_name_tweet_id,
                                          )
        # --------------------
        # If output_prob_mode is not None, get the token id of each token in `list_label_space`
        # --------------------
        if output_prob_mode:
            assert list_label_space is not None
            dict_label_space_token_ids = dict()
            list_label_space_first_token_id = []
            for label in list_label_space:
                # - {label: list of token ids}
                dict_label_space_token_ids[label] = self._get_token_id_from_word(label)
                list_label_space_first_token_id.append(dict_label_space_token_ids[label][0])
                tensor_label_space_first_token_id = th.tensor(list_label_space_first_token_id).to(self.device)
            # make sure that the first token of each label is unique
            assert len(set(list_label_space_first_token_id)) == len(list_label_space)
            # max_len_label_space_token_ids = max([len(x) for x in dict_label_space_token_ids.values()])
        # --------------------
        # iterate over the dataset and feed texts to the model
        # - https://huggingface.co/docs/datasets/v1.11.0/quicktour.html
        # multiple GPUs
        # - https://github.com/huggingface/transformers/issues/6821
        # --------------------
        # - collect the inputs and results
        # if keep_text:
        list_tweet_embedded = []
        if keep_tweet_id:
            list_tweet_id = []
        list_output_labels = []
        if output_prob_mode:
            list_output_label_probs = []
        # - specify the generator

        for i_batch, (input_ids, tweet_embedded, tweet_id, attention_mask) in enumerate(zip(pred_dataset["input_ids"],
                                                                                            pred_dataset["tweet_embedded"], pred_dataset[col_name_tweet_id],
                                                                                            pred_dataset["attention_mask"]
                                                                                            )):

            print("batch: {}/{}".format(i_batch, len(pred_dataset)))
            # # debug only
            # if i_batch < 62:
            #     continue
            # move the data to the device
            input_ids = input_ids.to(self.device)

            # generate the sentences
            if output_prob_mode == "label_space_first_token":
                # --------------------
                # generate with the log probabilities of each label in `list_label_space`
                # - based on the first token of each label
                # -- this ensure the length difference between labels will not affect the probabilities
                # --------------------
                label_space_first_token_log_prob = self._compute_log_prob_of_first_token_of_labels(input_ids, tensor_label_space_first_token_id)
                # -- normalize the log probabilities
                label_space_first_token_norm_log_prob = self._normalize_log_prob(label_space_first_token_log_prob)

                # --------------------
                # get the output labels along with the probabilities
                # --------------------
                outputs_labels = [list_label_space[i] for i in th.argmax(label_space_first_token_norm_log_prob, dim=1)]
                list_output_label_probs.append(label_space_first_token_norm_log_prob.to(th.float16).cpu().numpy())

            elif output_prob_mode == "label_space_all_tokens":
                # --------------------
                # generate with the log probabilities of each label in `list_label_space`
                # - based on the whole label
                # -- this may be problematic if different labels have different number of tokens, because the length difference will affect the probabilities. Therefore, in that case, we should only use the first token of each label.
                # --------------------
                label_space_complete_word_log_prob = self._compute_sum_log_prob_of_labels(input_ids, list_label_space, dict_label_space_token_ids, attention_mask)
                # -- normalize the log probabilities
                label_space_complete_word_norm_log_prob = self._normalize_log_prob(label_space_complete_word_log_prob)

                # --------------------
                # get the output labels along with the probabilities
                # --------------------
                outputs_labels = [list_label_space[i] for i in th.argmax(label_space_complete_word_norm_log_prob, dim=1)]
                list_output_label_probs.append(
                    label_space_complete_word_norm_log_prob.to(th.float16).detach().cpu().numpy())

            elif output_prob_mode == "generated_sequence":
                # --------------------
                # generated texts
                # - by default, it's using greedy search
                # -- therefore, temperature, top_k, top_p are not used
                # -- see
                # --- https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationMixin
                # --- more on generate with probabilities
                # --- https://huggingface.co/docs/transformers/model_doc/t5
                # --- https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14
                # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/23?page=2
                # --------------------
                # - if `do_sample` is False (default), `renormalize_logits` has no effect in the sequence , but will have an effect on the output scores
                # -- "the renormalize_logits parameter should not affect the selected tokens. However, the output scores themselves can still be affected by renormalize_logits. When renormalize_logits=True, the logits are renormalized using the specified temperature before they are passed to the softmax function. This renormalization step can lead to a change in the scale of the logits, which in turn can affect the output scores. However, the relative ranking of the logits (which token has the highest logit) remains the same, so the selected tokens during greedy search won't change. To summarize, the renormalize_logits parameter can affect the output scores, but it does not change the selected tokens during greedy search. If you are only interested in the generated tokens, you can safely ignore the difference in the output scores."
                outputs = self.model.generate(input_ids,
                                              output_scores=True, return_dict_in_generate=True,
                                              renormalize_logits=True)
                # - remove the first pad token (T5 uses the pad_token_id as the decoder_start_token_id)
                outputs_ids = outputs["sequences"][:, 1:]
                outputs_labels = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)

                # the probabilities of the generated tokens
                sum_log_probs = self._compute_sum_log_prob_of_generated_seq(outputs)
                assert NotImplementedError("Not implemented yet. Should save the sum_log_probs to the result df.")

            elif output_prob_mode is None:
                outputs = self.model.generate(input_ids)
                # print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
                outputs_labels = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            else:
                raise ValueError(f"output_prob_mode {output_prob_mode} not supported.")

            # if keep_text:
            list_tweet_embedded.append(tweet_embedded)
            if keep_tweet_id:
                list_tweet_id.append(tweet_id)
            list_output_labels.append(outputs_labels)
        # --------------------
        # create a df for the inputs and the predictions
        # --------------------
        # if keep_text:
        list_tweet_embedded_flat = flatten_1d_nested_list(list_tweet_embedded)
        if keep_tweet_id:
            list_tweet_id_flat = flatten_1d_nested_list(list_tweet_id)
        list_output_labels_flat = flatten_1d_nested_list(list_output_labels)
        if output_prob_mode:
            # (n_samples, n_labels)
            np_output_label_probs = np.concatenate(list_output_label_probs, axis=0)

        df_predictions = pd.DataFrame({self.col_name_label: list_output_labels_flat})
        if keep_tweet_id:
            df_predictions[self.col_name_text_id] = list_tweet_id_flat
        if keep_text:
            df_predictions[self.col_name_text] = list_tweet_embedded_flat
        else:
            # keep the top-10 rows of texts
            df_predictions[self.col_name_text] = list_tweet_embedded_flat[:10] + [""] * (len(list_tweet_embedded_flat) - 10)
        if output_prob_mode:
            for i, label in enumerate(list_label_space):
                df_predictions[f"{label}_norm_prob"] = np_output_label_probs[:, i]
        else:
            # only keep the top-10 rows of texts
            df_predictions[self.col_name_text] = list_tweet_embedded_flat[:10] + [""] * (len(list_tweet_embedded_flat) - 10)

        # --------------------
        # write to csv
        # --------------------
        creat_dir_for_a_file_if_not_exists(file_output_predictions)
        df_predictions.to_csv(file_output_predictions, index=False)

    def read_predicted_labels(self, file_output_predictions):
        """Read the predictions from the output file.

        Args:
            file_output_predictions (str): the path of the output predictions.
        """
        # check if the file exists
        if not os.path.exists(file_output_predictions):
            raise ValueError(f"The file {file_output_predictions} does not exist.")
        # read the file
        # - should read tweet_id and row_id as str
        df_predictions = pd.read_csv(file_output_predictions, dtype={self.col_name_text_id: str, "row_id": str})
        # - check if the columns are correct
        assert self.col_name_label in df_predictions.columns, f"The column {self.col_name_label} is not in the predictions."
        return df_predictions

    def _load_model(self):
        raise NotImplementedError()

    def _is_special_token(self, array_token_id):
        # return self.tokenizer.get_special_tokens_mask(array_token_id, already_has_special_tokens=True)
        return (array_token_id.unsqueeze(-1) == self.tensor_special_tokens).any(dim=-1)

    def _get_token_id_from_word(self, word):
        return self.tokenizer.encode(word, add_special_tokens=False)

    def _compute_log_prob_of_first_token_of_labels(self, input_ids, tensor_label_space_first_token_id):
        """
        Compute the log-probabilities of the first token of the labels in the label space. This is different from the `_compute_sum_log_prob_of_labels()` in that the latter computes the sum of log-probabilities of the labels. This method assumes that the first token of each labels in the label space is unique.
        Args:
            input_ids (torch.Tensor): the input ids of the texts in the prompt. (batch_size x len_input_tokens)
            tensor_label_space_first_token_id (torch.Tensor): a tensor of the first token id of the labels in the label space. E.g., [label_1_first_token, label_2_first_token]. Note that only the first token of each label is used. Also, the order of the label should be the same as the order in `list_label_space`. (num_label_spaces x 1)
        Returns:
            torch.Tensor: the sum of log-probabilities of the labels. (batch_size x num_label_spaces)

        Notes:
            - This should be equivalent to `_compute_sum_log_prob_of_labels()` when there is only one token per label in `dict_label_space_token_ids`. (empirically, I should set `renormalize_logits = False` in order to get the equivalent results. They may still not be exactly the same due to implementation differences and potential modifications applied during the generation process.)
            - https://github.com/ThilinaRajapakse/simpletransformers/issues/
            - https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationMixin.generate
        """
        batch_size, _ = input_ids.shape
        # --------------------
        # generate with the log probabilities of each label in `list_label_space`
        # --------------------
        # - (batch_size x 1)
        # -- [text_1_log_prob_1, text_1_log_prob_2, ...]
        # - note: `renormalize_logits=True` makes no difference
        # - note:  top_k = 0 makes no difference
        outputs = self.model.generate(input_ids,
                                      output_scores=True, return_dict_in_generate=True,
                                      max_new_tokens=1, renormalize_logits=True)
        # - get the logits first
        # -- (batch_size x num_tokens_in_dictionary])
        outputs_logits = outputs["scores"][0]
        # - convert the logits to the log-probabilities
        # -- (batch_size x num_tokens_in_dictionary])
        log_probs = logits_to_log_probs(outputs_logits)
        # get the log-probability of the token of interest
        # - (batch_size * 1)

        return log_probs[:, tensor_label_space_first_token_id]

    def _compute_sum_log_prob_of_labels(self, input_ids, list_label_space, dict_label_space_token_ids, attention_mask):
        """
        Compute the sum of log-probabilities of the labels in the label space.
        Args:
            input_ids (torch.Tensor): the input ids of the texts in the prompt. (batch_size x len_input_tokens)
            list_label_space (list): a list of possible labels.
            dict_label_space_token_ids (dict): a dict of the list of token ids of the labels in the label space. E.g., {'label_1': [token_id_1, token_id_2, ...], 'label_2': [token_id_1, token_id_2, ...], ...}
            attention_mask (torch.Tensor): the attention mask of the texts in the prompt (e.g., to ignore the padding tokens). (batch_size x len_input_tokens)

        Returns:
            torch.Tensor: the sum of log-probabilities of the labels. (batch_size x num_label_spaces)

        Notes:
            - use the `forward()` method instead of the `generate()` method. This allows "teacher forcing", which enforces the model to compute the logits for all tokens for any given label in the output label space.
                - https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Model.forward
                - https://stackoverflow.com/questions/73411215/getting-logits-from-t5-hugging-face-model-using-forward-method-without-labels
                - https://stackoverflow.com/questions/67328345/how-to-use-forward-method-instead-of-model-generate-for-t5-model
        """
        batch_size = input_ids.shape[0]
        # collect the log-probabilities of the labels in the label space
        tensor_label_space_log_prob = th.zeros((batch_size, len(list_label_space))).to(self.device)

        # - loop through the labels in the label space
        for i_label, (label_text, list_token_ids) in enumerate(dict_label_space_token_ids.items()):
            label_ids = th.tensor(list_token_ids).unsqueeze(0).to(self.device)
            # repeat the label_ids by the batch size
            label_ids = label_ids.repeat(batch_size, 1)
            with th.no_grad():
                outputs = self.model(input_ids=input_ids,
                                     labels=label_ids,
                                     attention_mask=attention_mask.to(self.device),
                                     use_cache=False)
            # - get the logits
            # -- (batch_size x len_seq x num_tokens_in_dictionary])
            # --- can move the logits to the CPU to save GPU memory
            logits = outputs.logits
            # - swap the len_seq and num_tokens_in_dictionary dimensions
            # -- (batch_size x num_tokens_in_dictionary x len_seq])
            logits = logits.permute(0, 2, 1)
            # - find the logits of the tokens of interest
            # -- (batch_size x len_seq])
            log_probs_path, sum_log_probs = index_logits_by_1d_index_tensor(logits, label_ids)

            # - add the log-probabilities of the labels to the tensor
            tensor_label_space_log_prob[:, i_label] = sum_log_probs
        return tensor_label_space_log_prob

    def _compute_sum_log_prob_of_generated_seq(self, outputs):
        """
        Compute the sum of log-probabilities of the generated sequence, excluding the special tokens.

        Args:
            outputs (dict): the outputs of the model.

        Returns:
            torch.Tensor: the sum of log-probabilities of the generated sequence. (batch_size)
        """
        assert "scores" in outputs
        assert "sequences" in outputs
        outputs_ids = outputs["sequences"][:, 1:]
        # - get the logits first
        # -- (batch_size x num_tokens_in_dictionary x len_output_tokens])
        outputs_logits = th.stack(outputs["scores"], dim=-1)
        batch_size, num_tokens_in_dictionary, len_output_tokens = outputs_logits.shape
        # - convert the logits to the log-probabilities
        # -- (batch_size x num_tokens_in_dictionary x len_output_tokens])
        log_probs = logits_to_log_probs(outputs_logits)

        # - sum the log-probabilities over the tokens of the output_labels
        # -- (batch_size x len_output_tokens)
        mask_special_token = self._is_special_token(outputs_ids)
        # - resize the mask for `log_probs`
        # -- (batch_size x num_tokens_in_dictionary x len_output_tokens)
        mask_special_token = mask_special_token.unsqueeze(1).expand_as(log_probs)
        # - zero out the log-probabilities of the special tokens
        log_probs_not_special = log_probs * (~mask_special_token)

        # - sum the log-probabilities over the tokens of the generated sequence
        log_probs_path, sum_log_probs = index_logits_by_1d_index_tensor(log_probs_not_special, outputs_ids)
        return sum_log_probs

    def _normalize_log_prob(self, tensor_log_prob):
        """Normalize the log-probabilities such that the sum of the output probabilities is 1.

        Args:
            tensor_log_prob (torch.Tensor): the log-probabilities. shape = (batch_size, num_labels)

        Returns:
            torch.Tensor: the normalized probabilities. The sum of the probabilities across all labels is 1. shape = (batch_size, num_labels)
        """
        assert tensor_log_prob.ndim == 2
        # convert the log-probabilities to the probabilities
        # - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        tensor_log_prob = th.exp(tensor_log_prob - tensor_log_prob.logsumexp(dim=-1).unsqueeze(-1))
        return tensor_log_prob


class FlanT5XxlLabelPredictor(FlanLabelPredictor):
    def __init__(self, col_name_text, col_name_label, col_name_text_id, try_use_gpu, mode_output="single-word", per_device_eval_batch_size=16, for_generation_only=True) -> None:
        """
        Args:
            col_name_text (str): the name of the column that contains the texts to be labeled.
            col_name_label (str): the name of the column that contains the predicted labels.
            col_name_text_id (str): the name of the column that contains the text id.
            try_use_gpu (bool): whether to try using GPU if available.
            mode_output (str): the mode of the output, e.g., "single-word".
            per_device_eval_batch_size (int): the batch size for each GPU. Default: 16.
            for_generation_only (bool): whether to use the model for generation only. If True, the model will use `device_map="auto"` when loading the model. This only matters if `try_use_gpu = True`. If False, the model will use `device_map=None` when loading the model, and then the model will be moved to the GPU using `model.parallelize()`. This has to do with the fact that, with GPU, model.forward() will have problems when the model is loaded using `device_map="auto"`.
                - see: https://discuss.pytorch.org/t/model-parallelize-runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least-two-devices-cuda-0-and-cuda-1/172673/2
        Notes:
            - https://huggingface.co/google/flan-t5-xxl
        """
        super().__init__(col_name_text, col_name_label, col_name_text_id, "flan-t5-xxl", try_use_gpu, per_device_eval_batch_size, mode_output)
        self.for_generation_only = for_generation_only

    def _load_model(self):
        """Load the model for labeling."""
        # - https://huggingface.co/google/flan-t5-xxl/discussions/9
        self.tokenizer = T5Tokenizer.from_pretrained("google/" + self.name_model)
        # if self.try_use_gpu:
        #     self.tokenizer = self.accelerator.prepare(self.tokenizer)

        if self.try_use_gpu:
            # self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
            #                                                         torch_dtype=th.bfloat16)
            # # self.model.to(self.device)
            # self.model = self.accelerator.prepare(self.model)

            if self.for_generation_only:
                device_map = "auto"
            else:
                device_map = None
            self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
                                                                    device_map=device_map,
                                                                    torch_dtype=th.bfloat16)
            if not self.for_generation_only:
                self.model.parallelize()
        else:
            self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
                                                                    torch_dtype=th.bfloat16)
        # load the special tokens
        self.tensor_special_tokens = th.tensor(
            [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id],
            dtype=th.long, device=self.device
        )


class FlanT5LargeLabelPredictor(FlanLabelPredictor):
    def __init__(self, col_name_text, col_name_label, col_name_text_id, try_use_gpu, mode_output="single-word", per_device_eval_batch_size=16, for_generation_only=True) -> None:
        """
        See the documentation of `FlanT5XxlLabelPredictor` for the meaning of the arguments.
        This class is for the large version of the Flan T5 model for working on Google Colab given the GPU memory limit.
        Notes:
            - https://huggingface.co/google/flan-t5-large
        """
        super().__init__(col_name_text, col_name_label, col_name_text_id, "flan-t5-large", try_use_gpu, per_device_eval_batch_size, mode_output)
        self.for_generation_only = for_generation_only

    def _load_model(self):
        """Load the model for labeling."""
        self.tokenizer = T5Tokenizer.from_pretrained("google/" + self.name_model)
        # if self.try_use_gpu:
        #     self.tokenizer = self.accelerator.prepare(self.tokenizer)

        if self.try_use_gpu:
            # self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
            #                                                         torch_dtype=th.bfloat16)
            # self.model.to(self.device)
            if self.for_generation_only:
                device_map = "auto"
            else:
                device_map = None
            self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
                                                                    device_map=device_map,
                                                                    torch_dtype=th.bfloat16)
            if not self.for_generation_only:
                self.model.parallelize()
        else:
            self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
                                                                    torch_dtype=th.bfloat16)
        # load the special tokens
        self.tensor_special_tokens = th.tensor(
            [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id],
            dtype=th.long, device=self.device
        )


class FlanT5SmallLabelPredictor(FlanLabelPredictor):
    def __init__(self, col_name_text, col_name_label, col_name_text_id, try_use_gpu, mode_output="single-word", per_device_eval_batch_size=16, for_generation_only=True) -> None:
        """
        See the documentation of `FlanT5XxlLabelPredictor` for the meaning of the arguments.
        This class is for the small version of the Flan T5 model and is for debugging purposes only to speed up.
        Notes:
            - https://huggingface.co/google/flan-t5-small
        """
        super().__init__(col_name_text, col_name_label, col_name_text_id, "flan-t5-small", try_use_gpu, per_device_eval_batch_size, mode_output)
        self.for_generation_only = for_generation_only

    def _load_model(self):
        """Load the model for labeling."""
        self.tokenizer = T5Tokenizer.from_pretrained("google/" + self.name_model)
        # if self.try_use_gpu:
        #     self.tokenizer = self.accelerator.prepare(self.tokenizer)

        if self.try_use_gpu:
            # self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
            #                                                         torch_dtype=th.bfloat16)
            # self.model.to(self.device)
            if self.for_generation_only:
                device_map = "auto"
            else:
                device_map = None
            self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
                                                                    device_map=device_map,
                                                                    torch_dtype=th.bfloat16)
            if not self.for_generation_only:
                self.model.parallelize()
        else:
            self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
                                                                    torch_dtype=th.bfloat16)
        # load the special tokens
        self.tensor_special_tokens = th.tensor(
            [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id],
            dtype=th.long, device=self.device
        )


class FlanUl2LabelPredictor(FlanLabelPredictor):
    def __init__(self, col_name_text, col_name_label, col_name_text_id, try_use_gpu, per_device_eval_batch_size=16, mode_output="single-word") -> None:
        """
        Args:
            col_name_text (str): the name of the column that contains the texts to be labeled.
            col_name_label (str): the name of the column that contains the predicted labels.
            col_name_text_id (str): the name of the column that contains the text id.
            try_use_gpu (bool): whether to try using GPU if available.
            per_device_eval_batch_size (int): the batch size for each GPU.
            mode_output (str): the mode of the output, e.g., "single-word".
        Notes:
            - https://huggingface.co/google/flan-ul2
        """
        super().__init__(col_name_text, col_name_label, col_name_text_id, "flan-ul2", try_use_gpu, per_device_eval_batch_size, mode_output)

    def _load_model(self):
        """Load the model for labeling."""
        self.tokenizer = T5Tokenizer.from_pretrained("google/" + self.name_model)
        if self.try_use_gpu:
            self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
                                                                    device_map="auto", torch_dtype=th.bfloat16)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained("google/" + self.name_model,
                                                                    torch_dtype=th.bfloat16)


if __name__ == "__main__":
    ts_start = datetime.datetime.now()
    print("--------------")
    print("start time: {}".format(ts_start))
    print("--------------")
    par = get_parameters_for_dataset()
    DATASET_META = par.DATASET_META
    # --------------------
    # Evaluate the GPT-generated labels
    # --------------------
    MODEL_TYPE = par.MODEL_GPT
    # MODEL_TYPE = "flan-t5-xxl"
    # MODEL_TYPE = "flan-ul2"
    # MODEL_TYPE = "gpt-chat-turbo-3_5"
    VERSION_PROMPT = par.VERSION_PROMPT
    # PARTITION_TYPE = "all"
    PARTITION_TYPE = "single_domain"
    # PARTITION_TYPE = "gda"

    # should adjust corresponding to the prompt type
    # MODE_OUTPUT = "single-word"
    MODE_OUTPUT = "CoT"

    # - whether to use the model for generation only (vs forward())
    FOR_GENERATION_ONLY = True

    # - should set to None if not debugging
    # - set to a small value if debugging
    LIMIT_FIRST_N = None
    # LIMIT_FIRST_N = 35

    # - the probability output along with the label
    # -- either None,"label_space_first_token","label_space_all_tokens","generated_sequence"
    # --- None means no probability output
    # --- "label_space_first_token" means the probability of the label in the label space of the first token
    # --- "label_space_all_tokens" means the probability of the label in the label space of all tokens
    # --- "generated_sequence" means the probability of the token in the generated sequence
    # OUTPUT_PROB_MODE = "label_space_all_tokens"
    OUTPUT_PROB_MODE = None

    # ------------------
    # SemEval specific
    # ------------------
    if DATASET_META == "SEM_EVAL":
        TOPIC = "Abortion"
        DATASET = "SEM_EVAL"
        FILE_INPUT_TEXT = par.DICT_FILE_DATA_SEM_EVAL_PROCESSED_EMBEDDED[DATASET]
        FILE_OUTPUT_PREDICTIONS = par.DICT_FILE_DATA_SEM_EVAL_PROCESSED_GPT_LABELS[DATASET]
        FILE_OUTPUT_PREDICTIONS = FILE_OUTPUT_PREDICTIONS.replace(".csv", "_{}.csv".format(TOPIC))
    else:
        raise ValueError("unknown dataset_meta: {}".format(DATASET_META))

    # other constants
    PER_DEVICE_EVAL_BATCH_SIZE = 16
    TRY_USE_GPU = True

    # gda-only
    if PARTITION_TYPE == "gda":
        LIST_TIME_DOMAIN = par.DICT_TIMES[par.TIME_DOMAIN_UNIT][DATASET],
        LIST_SOURCE_DOMAIN = par.DICT_SOURCE_DOMAIN[par.TIME_DOMAIN_UNIT][DATASET]
    elif PARTITION_TYPE != "gda":
        LIST_TIME_DOMAIN = None
        LIST_SOURCE_DOMAIN = None
    PARAMS = {
        # constant
        "dataset_meta": DATASET_META,
        "dataset": DATASET,
        "model_type": MODEL_TYPE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "try_use_gpu": TRY_USE_GPU,
        "for_generation_only": FOR_GENERATION_ONLY,
        "version_prompt": par.VERSION_PROMPT,
        "version_output": par.DICT_VERSION_OUTPUT["gpt"][MODEL_TYPE],
        "version_data": par.VERSION_DATA,
        # - record only
        "file_input_text": FILE_INPUT_TEXT,
        "file_output_predictions": FILE_OUTPUT_PREDICTIONS,
        "partition_type": PARTITION_TYPE,
        "mode_output": MODE_OUTPUT,
        "limit_first_n": LIMIT_FIRST_N,
        # "with_norm_prob": WITH_NORM_PROB,
        # - either None,"label_space_first_token","label_space_all_tokens","generated_sequence"
        "output_prob_mode": OUTPUT_PROB_MODE,
        # gda-only
        "list_time_domain": LIST_TIME_DOMAIN,
        "list_source_domain": LIST_SOURCE_DOMAIN
    }
    if DATASET_META == "COVID_VACCINE":
        PARAMS["task"] = TASK
    elif DATASET_META == "SEM_EVAL":
        PARAMS["topic"] = TOPIC

    main(PARAMS)

    ts_end = datetime.datetime.now()
    print("--------------")
    print("End time: {}".format(ts_end))
    print("Duration: {}".format(ts_end - ts_start))
    print("--------------")

    # --------------
    # 2*GPU, batch size = 16
    # End time: 2023-02-18 20:46:19.636563
    # Duration: 0:00:55.301771
    # --------------
    # --------------
    # 2*GPU, batch size = 32
    # End time: 2023-02-18 20:48:23.066206
    # Duration: 0:00:54.090863
    # --------------
