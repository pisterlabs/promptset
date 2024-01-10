# This script defines a class Ai Participant. When initialized and provided with a prompt,
# the Ai Participant sends a request to a ChatGPT model, obtains a response back and saves it
# in tabular data form.
# Sara Bonati - Center for Humans and Machines @ Max Planck Institute for Human Development
# ------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pandera as pa
import os
import json
import openai
import tiktoken
import itertools
from dotenv import load_dotenv
import argparse
import yaml
import re
from tqdm import tqdm
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
    wait_random_exponential,
)
from parsing_utils import extract_response


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


class AiParticipant:
    def __init__(self, game: str, model: str, n: int):
        """
        Initialzes AIParticipant object
        :param game: the economic game
        :param model: the openai model to use
        :param n: the number of times a prompt is sent for each parameter combination
        """

        # load environment variables
        load_dotenv()
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_ENDPOINT")
        openai.api_version = "2023-03-15-preview"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # This will correspond to the custom name you chose for your deployment when you deployed a model.
        assert model in [
            "gpt-35-turbo",
            "text-davinci-003",
        ], f"model needs to be either gpt-35-turbo or text-davinci-003, got {model} instead"
        self.deployment_name = model

        # load game specific params
        self.game = game
        self.game_params = load_yaml(f"./params/{game}.yml")

        # define OpenAI model to use during experiment
        self.model_code = model

        # define number of times a prompt needs to be sent for each combination
        self.n_control = 10
        # define number of times a prompt needs to be sent for each combination
        self.n = n

    @staticmethod
    def count_num_tokens(prompt: str, model_code: str):
        """
        This function counts the number of tokens a prompt is mad eof for different models.
        Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        :param prompt:
        :param model_code:
        :return:
        """

        model_dict = {"davinci": "text-davinci-003", "chatgpt": "gpt-3.5-turbo"}
        model = model_dict[model_code]
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "text-davinci-003":
            num_tokens = len(encoding.encode(prompt))
            return num_tokens

        if model == "gpt-3.5-turbo":
            num_tokens = 0
            messages = [{"role": "user", "content": prompt}]
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

    @staticmethod
    @retry(wait=wait_fixed(3) + wait_random(0, 2), stop=stop_after_attempt(4))
    def send_prompt(prompt: str, temp: float, model: str, game: str, mode: str):
        """
        This function sends a prompt to a specific language model with
        specific parameters.
        :param prompt: the prompt text
        :param temp: the temperature model parameter value to use
        :param model: the type of openai model to use in the API request
        :param game: the economic game
        :param mode: either control questions or running the experiment
        :return:
        """

        # add this to avoid rate limit errors
        # time.sleep(3)

        if model == "text-davinci-003":
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                temperature=temp,
                max_tokens=50,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            prompt_response = response["choices"][0]["text"].rstrip("\n")
            return prompt_response

        elif model == "gpt-35-turbo":
            response = openai.ChatCompletion.create(
                engine=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            prompt_response = response["choices"][0]["message"]["content"]
            print(prompt_response)
            finish_reason = response["choices"][0]["finish_reason"]
            # response returned as json
            prompt_response_dict = json.loads(prompt_response)

            if mode == "test_assumptions":
                return prompt_response_dict
            else:
                if game == "ultimatum_receiver":
                    prompt_response_value = prompt_response_dict["decision"]
                else:
                    prompt_response_value = prompt_response_dict["amount_sent"]

                return (
                    prompt_response_dict["reply"],
                    prompt_response_value,
                    finish_reason,
                )

    def adjust_prompts(self):
        """
        This function adjusts the prompts loaded from a csv file to include factors that
        may influence the AiParticipant response. Each factor is added to the prompt and the
        final prompt is saved in the same file in a new column

        :param prompt_file: (str) the name of the file containing prompts for the specific economic game
        :return:
        """

        # baseline prompts
        if self.game.endswith("_sender"):
            row_list_b = []
            for t in self.game_params["temperature"]:
                for i in range(self.n):
                    row = [self.game_params[f"prompt_baseline"]] + [t]
                    row_list_b.append(row)
            baseline_df = pd.DataFrame(
                row_list_b, columns=list(["prompt", "temperature"])
            )

        else:
            row_list_b = []
            factors_b = self.game_params["factors_baseline_names"]
            factors_list_b = [self.game_params[f] for f in factors_b]
            all_comb_b = list(itertools.product(*factors_list_b))
            print(f"Number of combinations (baseline): {len(all_comb_b)}")

            for t in self.game_params["temperature"]:
                for comb in all_comb_b:
                    for i in range(self.n):
                        prompt_params_dict = {
                            factors_b[f]: comb[f] for f in range(len(factors_b))
                        }

                        if self.game == "dictator_binary":
                            prompt_params_dict["fairness2"] = (
                                10 - prompt_params_dict["fairness"]
                            )

                        final_prompt = self.game_params[f"prompt_baseline"].format(
                            **prompt_params_dict
                        )
                        row = (
                            [final_prompt]
                            + [t]
                            + [prompt_params_dict[f] for f in factors_b]
                        )
                        row_list_b.append(row)
            baseline_df = pd.DataFrame(
                row_list_b, columns=list(["prompt", "temperature"] + factors_b)
            )

        # for the baseline prompts these two factors will be filled later
        baseline_df["age"] = "not_prompted"
        baseline_df["gender"] = "not_prompted"
        # add prompt type
        baseline_df["prompt_type"] = "baseline"
        baseline_df["game_type"] = self.game

        # add number of tokens that the prompt corresponds to
        # baseline_df["n_tokens_davinci"] = baseline_df["prompt"].apply(
        #     lambda x: self.count_num_tokens(x, "davinci")
        # )
        # baseline_df["n_tokens_chatgpt"] = baseline_df["prompt"].apply(
        #     lambda x: self.count_num_tokens(x, "chatgpt")
        # )

        # experimental prompts
        row_list_e = []
        factors_e = self.game_params["factors_experimental_names"]
        factors_list_e = [self.game_params[f] for f in factors_e]
        all_comb_e = list(itertools.product(*factors_list_e))
        print(f"Number of combinations (experimental): {len(all_comb_e)}")

        for t in self.game_params["temperature"]:
            for comb in all_comb_e:
                for i in range(self.n):
                    prompt_params_dict = {
                        factors_e[f]: comb[f] for f in range(len(factors_e))
                    }

                    if self.game == "dictator_binary":
                        prompt_params_dict["fairness2"] = (
                            10 - prompt_params_dict["fairness"]
                        )

                    final_prompt = self.game_params[f"prompt_complete"].format(
                        **prompt_params_dict
                    )
                    row = (
                        [final_prompt]
                        + [t]
                        + [prompt_params_dict[f] for f in factors_e]
                    )
                    row_list_e.append(row)

        experimental_df = pd.DataFrame(
            row_list_e, columns=list(["prompt", "temperature"] + factors_e)
        )
        experimental_df["prompt_type"] = "experimental"
        experimental_df["game_type"] = self.game

        # add number of tokens that the prompt corresponds to
        # experimental_df["n_tokens_davinci"] = experimental_df["prompt"].apply(
        #     lambda x: self.count_num_tokens(x, "davinci")
        # )
        # experimental_df["n_tokens_chatgpt"] = experimental_df["prompt"].apply(
        #     lambda x: self.count_num_tokens(x, "chatgpt")
        # )

        # pandera schema validation
        prompt_schema = pa.DataFrameSchema(
            {
                "prompt": pa.Column(str),
                "temperature": pa.Column(float, pa.Check.isin([0, 0.5, 1, 1.5])),
                "age": pa.Column(
                    str, pa.Check.isin(["not_prompted", "18-30", "31-50", "51-70"])
                ),
                "gender": pa.Column(
                    str, pa.Check.isin(["not_prompted", "female", "male", "non-binary"])
                ),
                "prompt_type": pa.Column(str),
                "game_type": pa.Column(str),
                # "n_tokens_davinci": pa.Column(int, pa.Check.between(0, 1000)),
                # "n_tokens_chatgpt": pa.Column(int, pa.Check.between(0, 1000)),
            }
        )

        try:
            prompt_schema.validate(baseline_df, lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)
        try:
            prompt_schema.validate(experimental_df, lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)

        # save both prompt types dataframes
        if not os.path.exists(os.path.join(prompts_dir, self.game)):
            os.makedirs(os.path.join(prompts_dir, self.game))
        baseline_df.to_csv(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_baseline_prompts_FIX.csv"
            )
        )
        experimental_df.to_csv(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_experimental_prompts_FIX.csv"
            )
        )

    def convert_data_to_jsonl(self, prompt_type: str):
        """
        This function converts the csv files with prompts and parameters into a JSONL file
        for async API calls. Each line in the JSONL corresponds to metadata to be sent with the API request

        :param prompt_type: the type of prompts (baseline or experimental)
        :return:
        """
        assert os.path.exists(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_{prompt_type}_prompts.csv"
            )
        ), "Prompts file does not exist"

        prompt_df = pd.read_csv(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_{prompt_type}_prompts.csv"
            )
        )

        JSON_file = []
        JSON_filename = f"{self.game}_{prompt_type}_prompts.jsonl"

        for index, row in prompt_df.iterrows():
            JSON_file.append(
                {
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "temperature": row["temperature"],
                    "max_tokens": 50,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                }
            )

        with open(os.path.join(prompts_dir, self.game, JSON_filename), "w") as outfile:
            for entry in JSON_file:
                json.dump(entry, outfile)
                outfile.write("\n")

    def collect_control_answers(self):
        """
        This method collects answers to control questions
        :return:
        """

        tqdm.pandas(desc="Control questions progress")

        questions = [self.game_params["control_questions"]] * self.n_control
        control_df = pd.DataFrame(questions, columns=["control_question"])
        print("control dataframe created!")
        # Apply the function to the column and create a DataFrame from the resulting dictionaries
        new_columns_df = pd.DataFrame.from_records(
            control_df["control_question"].progress_apply(
                self.send_prompt,
                args=(1, "gpt-35-turbo", self.game, "test_assumptions"),
            )
        )
        control_df[new_columns_df.columns] = new_columns_df
        print("answers collected!")

        # pandera schema validation
        # define schema
        if self.game == "dictator_sender" or self.game == "dictator_sequential":
            control_schema = pa.DataFrameSchema(
                {
                    "control_question": pa.Column(str),
                    "question1": pa.Column(int, coerce=True),
                    "question2": pa.Column(int, coerce=True),
                    "question3": pa.Column(int, coerce=True),
                    "question4": pa.Column(str),
                    "question5": pa.Column(str),
                }
            )

        elif self.game == "ultimatum_sender":
            control_schema = pa.DataFrameSchema(
                {
                    "control_question": pa.Column(str),
                    "question1": pa.Column(int, coerce=True),
                    "question2": pa.Column(int, coerce=True),
                    "question3": pa.Column(int, coerce=True),
                    "question4": pa.Column(str),
                    "question5": pa.Column(str),
                    "question6": pa.Column(int, coerce=True),
                    "question7": pa.Column(int, coerce=True),
                    "question8": pa.Column(int, coerce=True),
                    "question9": pa.Column(int, coerce=True),
                }
            )

        elif self.game == "ultimatum_receiver":
            control_schema = pa.DataFrameSchema(
                {
                    "control_question": pa.Column(str),
                    "question1": pa.Column(int, coerce=True),
                    "question2": pa.Column(int, coerce=True),
                    "question3": pa.Column(int, coerce=True),
                    "question4": pa.Column(int, coerce=True),
                    "question5": pa.Column(str),
                    "question6": pa.Column(str),
                    "question7": pa.Column(int, coerce=True),
                    "question8": pa.Column(int, coerce=True),
                    "question9": pa.Column(int, coerce=True),
                    "question10": pa.Column(int, coerce=True),
                }
            )

        try:
            control_schema.validate(control_df, lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)

        # save control questions and answers
        control_df.to_csv(
            os.path.join(prompts_dir, self.game, f"{self.game}_control_questions.csv"),
            sep=",",
            index=False,
        )

    def convert_jsonl_answers_to_csv(self, prompt_type: str):
        """
        This function converts the jsonl results file back into a csv file for later analysis.
        We extract responses from the JSON reply from ChatGPT + re-extract the factor levels for
        experimental prompts

        :param prompt_type: baseline or experimental
        :return:
        """

        results_jsonl = os.path.join(
            prompts_dir, self.game, f"{self.game}_{prompt_type}_results.jsonl"
        )
        assert os.path.exists(
            results_jsonl
        ), f"No results file found for {self.game} in {prompt_type} mode!"

        data = []
        count_lines_with_problems = 0
        ids_with_problems = []

        with open(results_jsonl) as f:
            for line in f:
                single_json_line = json.loads(line)

                if prompt_type == "baseline":
                    try:
                        response_dict = json.loads(
                            single_json_line[1]["choices"][0]["message"]["content"]
                        )
                        # print(single_json_line[1]["choices"][0]["message"]["content"])
                        # print(response_dict, any(isinstance(i,dict) for i in response_dict.values()) )
                        # print(list(response_dict.keys()))
                    except:
                        print(
                            f"Something went wrong (chat completion id: {single_json_line[1]['id']})"
                        )

                        ids_with_problems.append(single_json_line[1]["id"])

                        count_lines_with_problems += 1
                        if "content" in single_json_line[1]["choices"][0]["message"]:
                            print(
                                single_json_line[1]["choices"][0]["message"]["content"]
                            )
                        else:
                            print(single_json_line[1])

                    if (
                        self.game == "dictator_sender"
                        or self.game == "ultimatum_sender"
                    ):
                        if not any(isinstance(i, dict) for i in response_dict.values()):
                            data.append(
                                [
                                    single_json_line[0]["messages"][0][
                                        "content"
                                    ],  # the full prompt
                                    single_json_line[0]["temperature"],  # temperature
                                    "unprompted",  # age
                                    "unprompted",  # gender
                                    prompt_type,  # prompt_type
                                    self.game,  # game type,
                                    single_json_line[1]["id"],  # the completion id
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                                    ],
                                    # response_dict["reply"],  # the full response text
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "amount" in i or "money" in i or "sent" in i][0]}'
                                    ],
                                    # response_dict["amount_sent"],  # the integer reply
                                    single_json_line[1]["choices"][0][
                                        "finish_reason"
                                    ],  # finish reason
                                ]
                            )
                        else:
                            keys_2 = list(
                                response_dict[list(response_dict.keys())[0]].keys()
                            )
                            data.append(
                                [
                                    single_json_line[0]["messages"][0][
                                        "content"
                                    ],  # the full prompt
                                    single_json_line[0]["temperature"],  # temperature
                                    "unprompted",  # age
                                    "unprompted",  # gender
                                    prompt_type,  # prompt_type
                                    self.game,  # game type,
                                    single_json_line[1]["id"],  # the completion id
                                    response_dict[list(response_dict.keys())[0]][
                                        f'{[i for i in keys_2 if "re" in i or "Re" in i][0]}'
                                    ],
                                    # response_dict[list(response_dict.keys())[0]]["reply"],  # the full response text
                                    response_dict[list(response_dict.keys())[0]][
                                        f'{[i for i in keys_2 if "amount" in i or "money" in i or "sent" in i][0]}'
                                    ],
                                    # response_dict[list(response_dict.keys())[0]]["amount_sent"],  # the integer reply
                                    single_json_line[1]["choices"][0][
                                        "finish_reason"
                                    ],  # finish reason
                                ]
                            )

                    if (
                        self.game == "ultimatum_receiver"
                        or self.game == "dictator_sequential"
                    ):
                        if self.game == "dictator_sequential":
                            fairness = re.findall(
                                r"\d+",
                                single_json_line[0]["messages"][0]["content"].split(
                                    "."
                                )[5],
                            )
                            data.append(
                                [
                                    single_json_line[0]["messages"][0][
                                        "content"
                                    ],  # the full prompt
                                    single_json_line[0]["temperature"],  # temperature
                                    int(fairness[0]),  # fairness
                                    "unprompted",  # age
                                    "unprompted",  # gender
                                    prompt_type,  # prompt_type
                                    self.game,  # game type,
                                    single_json_line[1]["id"],  # the completion id
                                    # response_dict["reply"],  # the full response text
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                                    ],
                                    # response_dict["amount_sent"],  # the integer reply
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "amount" in i or "money" in i or "sent" in i][0]}'
                                    ],
                                    single_json_line[1]["choices"][0][
                                        "finish_reason"
                                    ],  # finish reason
                                ]
                            )
                        else:
                            fairness = re.findall(
                                r"\d+",
                                single_json_line[0]["messages"][0]["content"].split(
                                    "."
                                )[7],
                            )

                            if not any(
                                isinstance(i, dict) for i in response_dict.values()
                            ):
                                data.append(
                                    [
                                        single_json_line[0]["messages"][0][
                                            "content"
                                        ],  # the full prompt
                                        single_json_line[0][
                                            "temperature"
                                        ],  # temperature
                                        int(fairness[0]),  # fairness
                                        "unprompted",  # age
                                        "unprompted",  # gender
                                        prompt_type,  # prompt_type
                                        self.game,  # game type,
                                        single_json_line[1]["id"],  # the completion id
                                        response_dict[
                                            f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                                        ],
                                        # response_dict["reply"],  # the full response text
                                        response_dict[
                                            f'{[i for i in list(response_dict.keys()) if "decision" in i or "Decision" in i or "suggestion" in i or "dicision" in i][0]}'
                                        ],
                                        # response_dict["decision"],  # the integer reply
                                        single_json_line[1]["choices"][0][
                                            "finish_reason"
                                        ],  # finish reason
                                    ]
                                )

                            else:
                                keys_2 = list(
                                    response_dict[list(response_dict.keys())[0]].keys()
                                )
                                data.append(
                                    [
                                        single_json_line[0]["messages"][0][
                                            "content"
                                        ],  # the full prompt
                                        single_json_line[0][
                                            "temperature"
                                        ],  # temperature
                                        int(fairness[0]),  # fairness
                                        "unprompted",  # age
                                        "unprompted",  # gender
                                        prompt_type,  # prompt_type
                                        self.game,  # game type,
                                        single_json_line[1]["id"],  # the completion id
                                        response_dict[list(response_dict.keys())[0]][
                                            f'{[i for i in keys_2 if "re" in i or "Re" in i][0]}'
                                        ],
                                        # response_dict["reply"],  # the full response text
                                        response_dict[list(response_dict.keys())[0]][
                                            f'{[i for i in keys_2 if "decision" in i or "Decision" in i or "suggestion" in i][0]}'
                                        ],
                                        # response_dict["decision"],  # the integer reply
                                        single_json_line[1]["choices"][0][
                                            "finish_reason"
                                        ],  # finish reason
                                    ]
                                )

                    if self.game == "dictator_binary":
                        fairness = re.findall(
                            r"\d+",
                            single_json_line[0]["messages"][0]["content"].split(".")[3],
                        )

                        print(list(response_dict.keys()))
                        if len(list(response_dict.keys())) == 1:
                            print(response_dict)

                            if list(response_dict.keys())[0] == "data":
                                reply = response_dict["data"]["reply"]
                                option = response_dict["data"]["option_preferred"]
                            elif list(response_dict.keys())[0] == "reply":
                                if "option_preferred" in response_dict["reply"]:
                                    index = response_dict["reply"].find(
                                        "option_preferred"
                                    )
                                    reply = response_dict["reply"]
                                    option = re.findall(
                                        r"\d+", response_dict["reply"][index:]
                                    )[0]
                                    print(option)

                            data.append(
                                [
                                    single_json_line[0]["messages"][0][
                                        "content"
                                    ],  # the full prompt
                                    single_json_line[0]["temperature"],  # temperature
                                    int(fairness[0]),  # fairness
                                    "unprompted",  # age
                                    "unprompted",  # gender
                                    prompt_type,  # prompt_type
                                    self.game,  # game type,
                                    single_json_line[1]["id"],  # the completion id
                                    reply,
                                    option,
                                    single_json_line[1]["choices"][0][
                                        "finish_reason"
                                    ],  # finish reason
                                ]
                            )

                        else:
                            data.append(
                                [
                                    single_json_line[0]["messages"][0][
                                        "content"
                                    ],  # the full prompt
                                    single_json_line[0]["temperature"],  # temperature
                                    int(fairness[0]),  # fairness
                                    "unprompted",  # age
                                    "unprompted",  # gender
                                    prompt_type,  # prompt_type
                                    self.game,  # game type,
                                    single_json_line[1]["id"],  # the completion id
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                                    ],
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "option" in i][0]}'
                                    ],
                                    single_json_line[1]["choices"][0][
                                        "finish_reason"
                                    ],  # finish reason
                                ]
                            )

                elif prompt_type == "experimental":
                    try:
                        response_dict = json.loads(
                            single_json_line[1]["choices"][0]["message"]["content"]
                        )

                        # extract_response(single_json_line, self.game, prompt_type)
                        # print(single_json_line[1]["choices"][0]["message"]["content"])

                        message = single_json_line[0]["messages"][0]["content"]
                        demographic_phrase = message.split(". ")[1]
                        demographic_words = demographic_phrase.split(" ")
                        # collect demogrpahic info of each prompt
                        age = demographic_words[2]
                        gender = demographic_words[-1]

                    except:
                        print(
                            f"Something went wrong (chat completion id: {single_json_line[1]['id']})"
                        )

                        ids_with_problems.append(single_json_line[1]["id"])
                        count_lines_with_problems += 1

                    if self.game == "dictator_binary":
                        fairness = re.findall(
                            r"\d+",
                            single_json_line[0]["messages"][0]["content"].split(".")[4],
                        )

                        print(list(response_dict.keys()))
                        if len(list(response_dict.keys())) == 1:
                            print(response_dict)

                            if list(response_dict.keys())[0] == "data":
                                reply = response_dict["data"]["reply"]
                                option = response_dict["data"]["option_preferred"]
                            elif list(response_dict.keys())[0] == "reply":
                                if "option_preferred" in response_dict["reply"]:
                                    index = response_dict["reply"].find(
                                        "option_preferred"
                                    )
                                    reply = response_dict["reply"]
                                    option = re.findall(
                                        r"\d+", response_dict["reply"][index:]
                                    )[0]
                                    print(option)

                            data.append(
                                [
                                    single_json_line[0]["messages"][0][
                                        "content"
                                    ],  # the full prompt
                                    single_json_line[0]["temperature"],  # temperature
                                    int(fairness[0]),  # fairness
                                    age,  # age
                                    gender,  # gender
                                    prompt_type,  # prompt_type
                                    self.game,  # game type,
                                    single_json_line[1]["id"],  # the completion id
                                    reply,
                                    option,
                                    single_json_line[1]["choices"][0][
                                        "finish_reason"
                                    ],  # finish reason
                                ]
                            )

                        else:
                            data.append(
                                [
                                    single_json_line[0]["messages"][0][
                                        "content"
                                    ],  # the full prompt
                                    single_json_line[0]["temperature"],  # temperature
                                    int(fairness[0]),  # fairness
                                    age,  # age
                                    gender,  # gender
                                    prompt_type,  # prompt_type
                                    self.game,  # game type,
                                    single_json_line[1]["id"],  # the completion id
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                                    ],
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "option" in i or "opinion" in i or "preferred" in i or "expression" in i][0]}'
                                    ],
                                    single_json_line[1]["choices"][0][
                                        "finish_reason"
                                    ],  # finish reason
                                ]
                            )

                    if (
                        self.game == "dictator_sender"
                        or self.game == "ultimatum_sender"
                    ):
                        print(response_dict)
                        if not any(isinstance(i, dict) for i in response_dict.values()):
                            data.append(
                                [
                                    single_json_line[0]["messages"][0][
                                        "content"
                                    ],  # the full prompt
                                    single_json_line[0]["temperature"],  # temperature
                                    age,  # age
                                    gender,  # gender
                                    prompt_type,  # prompt_type
                                    self.game,  # game type,
                                    single_json_line[1]["id"],  # the completion id
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                                    ],
                                    # response_dict["reply"],  # the full response text
                                    response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "amount" in i or "money" in i or "sent" in i or "Sent" in i or "suggestion" in i][0]}'
                                    ],
                                    single_json_line[1]["choices"][0][
                                        "finish_reason"
                                    ],  # finish reason
                                ]
                            )

                        else:
                            print(list(response_dict.keys()))
                            if isinstance(
                                response_dict[list(response_dict.keys())[0]], dict
                            ):
                                keys_2_index = 0
                                keys_2 = list(
                                    response_dict[list(response_dict.keys())[0]].keys()
                                )
                            elif isinstance(
                                response_dict[list(response_dict.keys())[1]], dict
                            ):
                                keys_2_index = 1
                                keys_2 = list(
                                    response_dict[list(response_dict.keys())[1]].keys()
                                )
                            print(keys_2, keys_2_index)
                            print(
                                response_dict[list(response_dict.keys())[keys_2_index]]
                            )

                            # we have two edge cases
                            # 1) {'reply':.., 'answer':{'amount_sent':..}} or {'reply':.., 'suggestion':{'amount_sent':..}}
                            # 2) {'answer': {'reply': ..., 'amount_sent': ...}}

                            if len(keys_2) == 1:
                                reply = response_dict[
                                    f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                                ]
                                data.append(
                                    [
                                        single_json_line[0]["messages"][0][
                                            "content"
                                        ],  # the full prompt
                                        single_json_line[0][
                                            "temperature"
                                        ],  # temperature
                                        age,  # age
                                        gender,  # gender
                                        prompt_type,  # prompt_type
                                        self.game,  # game type,
                                        single_json_line[1]["id"],  # the completion id
                                        reply,
                                        response_dict[
                                            list(response_dict.keys())[keys_2_index]
                                        ][
                                            f'{[i for i in keys_2 if "amount" in i or "money" in i or "sent" in i or "Sent" in i][0]}'
                                        ],
                                        single_json_line[1]["choices"][0][
                                            "finish_reason"
                                        ],  # finish reason
                                    ]
                                )

                            elif len(keys_2) == 2:
                                reply = response_dict[list(response_dict.keys())[0]][
                                    f'{[i for i in keys_2 if "re" in i or "Re" in i][0]}'
                                ]
                                data.append(
                                    [
                                        single_json_line[0]["messages"][0][
                                            "content"
                                        ],  # the full prompt
                                        single_json_line[0][
                                            "temperature"
                                        ],  # temperature
                                        age,  # age
                                        gender,  # gender
                                        prompt_type,  # prompt_type
                                        self.game,  # game type,
                                        single_json_line[1]["id"],  # the completion id
                                        reply,
                                        response_dict[list(response_dict.keys())[0]][
                                            f'{[i for i in keys_2 if "amount" in i or "money" in i or "sent" in i or "Sent" in i][0]}'
                                        ],
                                        single_json_line[1]["choices"][0][
                                            "finish_reason"
                                        ],  # finish reason
                                    ]
                                )

                    if (
                        self.game == "ultimatum_receiver"
                        or self.game == "dictator_sequential"
                    ):
                        if self.game == "dictator_sequential":
                            fairness = re.findall(
                                r"\d+",
                                single_json_line[0]["messages"][0]["content"].split(
                                    "."
                                )[6],
                            )

                            if not any(
                                isinstance(i, dict) for i in response_dict.values()
                            ):
                                print(response_dict)
                                if len(response_dict) > 1:
                                    data.append(
                                        [
                                            single_json_line[0]["messages"][0][
                                                "content"
                                            ],  # the full prompt
                                            single_json_line[0][
                                                "temperature"
                                            ],  # temperature
                                            int(fairness[0]),  # fairness
                                            age,  # age
                                            gender,  # gender
                                            prompt_type,  # prompt_type
                                            self.game,  # game type,
                                            single_json_line[1][
                                                "id"
                                            ],  # the completion id
                                            # response_dict["reply"],  # the full response text
                                            response_dict[
                                                f'{[i for i in list(response_dict.keys()) if "reply" in i or "Reply" in i or "reponse" in i or "response" in i][0]}'
                                            ],
                                            response_dict[
                                                f'{[i for i in list(response_dict.keys()) if "amount" in i or "money" in i or "sent" in i or "Sent" in i or "suggestion" in i][0]}'
                                            ],
                                            single_json_line[1]["choices"][0][
                                                "finish_reason"
                                            ],  # finish reason
                                        ]
                                    )
                                else:
                                    ids_with_problems.append(single_json_line[1]["id"])

                            else:
                                # for edge cases
                                if isinstance(
                                    response_dict[list(response_dict.keys())[0]], dict
                                ):
                                    keys_2_index = 0
                                    keys_2 = list(
                                        response_dict[
                                            list(response_dict.keys())[0]
                                        ].keys()
                                    )
                                elif isinstance(
                                    response_dict[list(response_dict.keys())[1]], dict
                                ):
                                    keys_2_index = 1
                                    keys_2 = list(
                                        response_dict[
                                            list(response_dict.keys())[1]
                                        ].keys()
                                    )
                                print(keys_2, keys_2_index)
                                print(
                                    response_dict[
                                        list(response_dict.keys())[keys_2_index]
                                    ]
                                )

                                if len(keys_2) == 1:
                                    reply = response_dict[
                                        f'{[i for i in list(response_dict.keys()) if "re" in i or "Re" in i][0]}'
                                    ]

                                    data.append(
                                        [
                                            single_json_line[0]["messages"][0][
                                                "content"
                                            ],  # the full prompt
                                            single_json_line[0][
                                                "temperature"
                                            ],  # temperature
                                            int(fairness[0]),  # fairness
                                            age,  # age
                                            gender,  # gender
                                            prompt_type,  # prompt_type
                                            self.game,  # game type,
                                            single_json_line[1][
                                                "id"
                                            ],  # the completion id
                                            reply,
                                            response_dict[
                                                list(response_dict.keys())[keys_2_index]
                                            ][
                                                f'{[i for i in keys_2 if "amount" in i or "money" in i or "sent" in i or "Sent" in i][0]}'
                                            ],
                                            single_json_line[1]["choices"][0][
                                                "finish_reason"
                                            ],  # finish reason
                                        ]
                                    )
                                elif len(keys_2) == 2:
                                    reply = response_dict[
                                        list(response_dict.keys())[0]
                                    ][
                                        f'{[i for i in keys_2 if "re" in i or "Re" in i][0]}'
                                    ]

                                    data.append(
                                        [
                                            single_json_line[0]["messages"][0][
                                                "content"
                                            ],  # the full prompt
                                            single_json_line[0][
                                                "temperature"
                                            ],  # temperature
                                            age,  # age
                                            gender,  # gender
                                            prompt_type,  # prompt_type
                                            self.game,  # game type,
                                            single_json_line[1][
                                                "id"
                                            ],  # the completion id
                                            reply,
                                            response_dict[
                                                list(response_dict.keys())[0]
                                            ][
                                                f'{[i for i in keys_2 if "amount" in i or "money" in i or "sent" in i or "Sent" in i][0]}'
                                            ],
                                            single_json_line[1]["choices"][0][
                                                "finish_reason"
                                            ],  # finish reason
                                        ]
                                    )

                        else:
                            fairness = re.findall(
                                r"\d+",
                                single_json_line[0]["messages"][0]["content"].split(
                                    "."
                                )[8],
                            )

                            if not any(
                                isinstance(i, dict) for i in response_dict.values()
                            ):
                                print(response_dict)

                                # for edge case {'response': [{'reply': 'Yes, you should accept.'}, {'decision': 1}]}
                                # and edge case {'response': [{'reply': 'Yes you should accept', 'decision': 1}]}
                                if any(
                                    isinstance(i, list) for i in response_dict.values()
                                ):
                                    if len(response_dict["response"]) > 1:
                                        data.append(
                                            [
                                                single_json_line[0]["messages"][0][
                                                    "content"
                                                ],  # the full prompt
                                                single_json_line[0][
                                                    "temperature"
                                                ],  # temperature
                                                int(fairness[0]),  # fairness
                                                age,  # age
                                                gender,  # gender
                                                prompt_type,  # prompt_type
                                                self.game,  # game type,
                                                single_json_line[1][
                                                    "id"
                                                ],  # the completion id
                                                # response_dict["reply"],  # the full response text
                                                response_dict["response"][0]["reply"],
                                                response_dict["response"][1][
                                                    "decision"
                                                ],
                                                single_json_line[1]["choices"][0][
                                                    "finish_reason"
                                                ],  # finish reason
                                            ]
                                        )
                                    else:
                                        data.append(
                                            [
                                                single_json_line[0]["messages"][0][
                                                    "content"
                                                ],  # the full prompt
                                                single_json_line[0][
                                                    "temperature"
                                                ],  # temperature
                                                int(fairness[0]),  # fairness
                                                age,  # age
                                                gender,  # gender
                                                prompt_type,  # prompt_type
                                                self.game,  # game type,
                                                single_json_line[1][
                                                    "id"
                                                ],  # the completion id
                                                # response_dict["reply"],  # the full response text
                                                response_dict["response"][0]["reply"],
                                                response_dict["response"][0][
                                                    "decision"
                                                ],
                                                single_json_line[1]["choices"][0][
                                                    "finish_reason"
                                                ],  # finish reason
                                            ]
                                        )

                                else:
                                    data.append(
                                        [
                                            single_json_line[0]["messages"][0][
                                                "content"
                                            ],  # the full prompt
                                            single_json_line[0][
                                                "temperature"
                                            ],  # temperature
                                            int(fairness[0]),  # fairness
                                            age,  # age
                                            gender,  # gender
                                            prompt_type,  # prompt_type
                                            self.game,  # game type,
                                            single_json_line[1][
                                                "id"
                                            ],  # the completion id
                                            # response_dict["reply"],  # the full response text
                                            response_dict[
                                                f'{[i for i in list(response_dict.keys()) if "re" in i or "stringResponse" in i or "Reply" in i][0]}'
                                            ],
                                            response_dict[
                                                f'{[i for i in list(response_dict.keys()) if "decision" in i or "appraisal" in i or "suggestion" in i or "descision" in i or "result" in i or "decison" in i or "Decision" in i][0]}'
                                            ],  # the integer reply
                                            single_json_line[1]["choices"][0][
                                                "finish_reason"
                                            ],  # finish reason
                                        ]
                                    )

        print(
            "Number of responses with problems in json parsing: ",
            count_lines_with_problems,
        )

        # save completion ids with problems (to be checked manually)
        with open(
            os.path.join(
                prompts_dir,
                self.game,
                f"{self.game}_{prompt_type}_completion_ids_to_check.txt",
            ),
            "w",
        ) as f:
            for line in ids_with_problems:
                f.write(f"{line}\n")

        print(f"Number of columns present in the dataset: {len(data[0])}")

        if self.game in ["dictator_sender", "ultimatum_sender"]:
            results_df = pd.DataFrame(
                data,
                columns=[
                    "prompt",
                    "temperature",
                    "age",
                    "gender",
                    "prompt_type",
                    "game",
                    "completion_id",
                    "reply_text",
                    "value",
                    "finish_reason",
                ],
            )

            # new! exclude rows when the field is empty and when the number in the field does not make sense.
            # (this exclusion is game specific)
            if self.game == "dictator_sender":
                # response at index 25541 in the jsonl file has /r character that broke response in two parts,
                # adjust here
                results_df = results_df[~results_df["completion_id"].isnull()]
                results_df.iloc[25534, -2] = 5
                results_df.iloc[25534, -1] = "stop"

                results_df["value"] = pd.to_numeric(
                    results_df["value"], errors="coerce"
                )
                results_df = results_df[
                    results_df["value"].between(0, 10, inclusive="both")
                ]
            if self.game == "ultimatum_sender":
                results_df["value"] = pd.to_numeric(
                    results_df["value"], errors="coerce"
                )
                results_df = results_df[
                    results_df["value"].between(0, 10, inclusive="both")
                ]

        else:
            results_df = pd.DataFrame(
                data,
                columns=[
                    "prompt",
                    "temperature",
                    "fairness",
                    "age",
                    "gender",
                    "prompt_type",
                    "game",
                    "completion_id",
                    "reply_text",
                    "value",
                    "finish_reason",
                ],
            )

            # new! exclude rows when the field is empty and when the number in the field does not make sense.
            # (this exclusion is game specific)
            if self.game == "dictator_binary":
                results_df["value"] = pd.to_numeric(
                    results_df["value"], errors="coerce"
                )
                results_df = results_df[results_df["value"].isin([1, 2, 1.0, 2.0])]
            if self.game == "ultimatum_receiver":
                results_df["value"] = pd.to_numeric(
                    results_df["value"], errors="coerce"
                )
                results_df = results_df[results_df["value"].isin([1, 0, 1.0, 0.0])]
            if self.game == "dictator_sequential":
                results_df["value"] = pd.to_numeric(
                    results_df["value"], errors="coerce"
                )
                results_df = results_df[
                    results_df["value"].between(0, 10, inclusive="both")
                ]

        # save results as csv
        results_df.to_csv(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_{prompt_type}_results_FIX.csv"
            ),
            sep=",",
        )

    def try_parsing(self, prompt_type: str):
        results_jsonl = os.path.join(
            prompts_dir, self.game, f"{self.game}_{prompt_type}_results.jsonl"
        )
        assert os.path.exists(
            results_jsonl
        ), f"No results file found for {self.game} in {prompt_type} mode!"

        data = []
        exceptions = []
        count_lines_with_problems = 0
        i = 0
        with open(results_jsonl) as f:
            for line in f:
                single_json_line = json.loads(line)
                result = extract_response(single_json_line, self.game, prompt_type)

                print(result)
                i += 1
                if i > 5:
                    break
                # if len(result) == 3:
                #    exceptions.append(result)
                # else:
                #    data.append(result)

    def collect_answers(self, prompt_type: str):
        """
        This function sends a request to an openai language model,
        collects the response and saves it in csv file

        :param prompt_type: baseline or experimental
        :return:
        """
        assert os.path.exists(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_{prompt_type}_prompts.csv"
            )
        ), "Prompts file does not exist"

        prompt_df = pd.read_csv(
            os.path.join(
                prompts_dir, self.game, f"{self.game}_{prompt_type}_prompts.csv"
            )
        )

        # tqdm.pandas(desc="Experiment progress")
        #
        # # retrieve answer text
        # prompt_df[
        #     ["answer_text", "answer", "finish_reason"]
        # ] = prompt_df.progress_apply(
        #     lambda row: self.send_prompt(
        #         row["prompt"], row["temperature"], self.model_code, self.game, ""
        #     ),
        #     axis=1,
        # ).apply(
        #     pd.Series
        # )

        for i in tqdm(range(len(prompt_df))):
            if i % 5 == 0:
                time.sleep(10)
            answer_text, answer, finish_reason = self.send_prompt(
                prompt_df.loc[i, "prompt"],
                prompt_df.loc[i, "temperature"],
                self.model_code,
                self.game,
                "",
            )
            prompt_df.loc[i, "answer_text"] = answer_text
            prompt_df.loc[i, "answer"] = answer
            prompt_df.loc[i, "finish_reason"] = finish_reason

        # pandera schema validation
        final_prompt_schema = pa.DataFrameSchema(
            {
                "prompt": pa.Column(str),
                "temperature": pa.Column(float, pa.Check.isin([0, 0.5, 1, 1.5])),
                "age": pa.Column(
                    str, pa.Check.isin(["not_prompted", "18-30", "31-50", "51-70"])
                ),
                "gender": pa.Column(
                    str, pa.Check.isin(["not_prompted", "female", "male", "non-binary"])
                ),
                "prompt_type": pa.Column(str),
                "game_type": pa.Column(str),
                "answer_text": pa.Column(str),
                "answer": pa.Column(int, pa.Check.between(0, 10), coerce=True),
                "finish_reason": pa.Column(
                    str, pa.Check.isin(["stop", "content_filter", "null", "length"])
                ),
            }
        )

        try:
            final_prompt_schema.validate(prompt_df, lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)

        prompt_df.to_csv(
            os.path.join(prompts_dir, self.game, f"{self.game}_data.csv"), index=False
        )


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="""
                                                 Run ChatGPT AI participant for economic games
                                                 (Center for Humans and Machines, 
                                                 Max Planck Institute for Human Development)
                                                 """
    )
    parser.add_argument(
        "--game",
        type=str,
        help="""
                                Which economic game do you want to run \n 
                                (options: dictator_sender, 
                                          dictator_sequential, 
                                          dictator_binary, 
                                          ultimatum_sender, 
                                          ultimatum_receiver)
                                """,
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="""
                                What type of action do you want to run \n 
                                (options: test_assumptions, 
                                          prepare_prompts,
                                          convert_prompts_to_json,
                                          send_prompts_baseline, 
                                          send_prompts_experimental,
                                          convert_jsonl_results_to_csv)
                                """,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="""
                                Which model do you want to use \n 
                                (options: gpt-35-turbo, text-davinci-003)
                                """,
        required=True,
    )

    # collect args and make assert tests
    args = parser.parse_args()
    general_params = load_yaml("params/params.yml")
    assert (
        args.game in general_params["game_options"]
    ), f'Game option must be one of {general_params["game_options"]}'
    assert (
        args.mode in general_params["mode_options"]
    ), f'Mode option must be one of {general_params["mode_options"]}'
    assert (
        args.model in general_params["model_options"]
    ), f'Model option must be one of {general_params["model_options"]}'

    # directory structure
    prompts_dir = "../prompts"

    # initialize AiParticipant object
    P = AiParticipant(args.game, args.model, general_params["n"])

    if args.mode == "test_assumptions":
        P.collect_control_answers()

    if args.mode == "prepare_prompts":
        P.adjust_prompts()

    if args.mode == "convert_prompts_to_json":
        # P.convert_data_to_jsonl("baseline")
        P.convert_data_to_jsonl("experimental")

    if args.mode == "send_prompts_baseline":
        P.collect_answers("baseline")

    if args.mode == "send_prompts_experimental":
        P.collect_answers("experimental")

    if args.mode == "convert_jsonl_results_to_csv":
        # P.convert_jsonl_answers_to_csv("baseline")
        P.convert_jsonl_answers_to_csv("experimental")
