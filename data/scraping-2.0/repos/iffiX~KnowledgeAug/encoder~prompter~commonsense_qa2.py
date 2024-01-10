import os
import re
import sys
import logging
import openai
from itertools import chain
from encoder.dataset.commonsense_qa2 import CommonsenseQA2BaseDataset
from encoder.utils.file import JSONStreamCache
from encoder.utils.settings import preprocess_cache_dir
from .prompts import prompts


class CommonsenseQA2Prompter:
    BASE_PROMPT = prompts["commonsense_qa2"]
    FORMATTED_PROMPT_EXPECTED_TOKENS = 500
    MODEL = "text-davinci-002"
    MODEL_PRICE_PER_THOUSAND_TOKENS = 0.02

    def __init__(
        self, dataset: CommonsenseQA2BaseDataset,
    ):
        self.dataset = dataset
        self.generation_table = {
            data["id"]: data
            for data in chain(
                self.dataset.train_data,
                self.dataset.validate_data,
                self.dataset.test_data,
            )
        }

        if not os.path.exists(
            os.path.join(preprocess_cache_dir, "commonsense_qa2_prompts.json")
        ):
            self.get_user_confirmation()

        with JSONStreamCache(
            os.path.join(preprocess_cache_dir, "commonsense_qa2_prompts.json"),
            [
                data["id"]
                for data in chain(
                    self.dataset.train_data,
                    self.dataset.validate_data,
                    self.dataset.test_data,
                )
            ],
            self.generator,
        ) as cache:
            self.search_hints = self.parse_data(cache.data)

    def get_search_hints_of_id(self, id_) -> list:
        return self.search_hints.get(id_, []) or [{"search": [], "starts": []}]

    def parse_data(self, raw_data):
        """
        raw_data example:
        {
            "8-791": [
                {
                  "text": "\n\nThis is a test",
                  "index": 0,
                  "logprobs": null,
                  "finish_reason": "length"
                }
            ]
        }

        Return:
        {
            "8-791": {
                "search": ["September", "days"],
                "starts": ["September"]
            }
        }
        """
        parsed_data = {}
        failed_to_parse_num = 0
        for id_, choices in raw_data.items():
            search_hints = []
            for choice in choices:
                try:
                    raw_text = choice["text"]
                    search_hint = {"search": [], "starts": []}

                    if "Search:" not in raw_text or (
                        "Starts:" not in raw_text and "Start:" not in raw_text
                    ):
                        logging.info(
                            f"id={id_}, input={raw_text}\n "
                            f"Error: failed to match fields"
                        )
                        raise ValueError()
                    search_start = raw_text.find("Search:") + len("Search:")
                    if "Starts:" in raw_text:
                        search_end = raw_text.find("Starts:")
                        starts_start = raw_text.find("Starts:") + len("Starts:")
                    else:
                        search_end = raw_text.find("Start:")
                        starts_start = raw_text.find("Start:") + len("Start:")
                    if search_start >= starts_start:
                        logging.info(
                            f"id={id_}, input={raw_text}\n "
                            f"Error: field order incorrect"
                        )
                        raise ValueError()

                    search_result = re.findall(
                        r"<([\w,\-/' \$\"&\.’%]+)>", raw_text[search_start:search_end],
                    )
                    if search_result is None:
                        logging.info(
                            f"id={id_}, input={raw_text}\n "
                            f"Error: search targets empty"
                        )
                        raise ValueError()

                    for se_r in search_result:
                        search_hint["search"].append(
                            [x.strip(" ") for x in se_r.split(",")]
                        )

                    starts_result = re.findall(
                        r"\[([\w,\-/' \$\"&\.’%]+)\]", raw_text[starts_start:],
                    )
                    if starts_result is None:
                        logging.info(
                            f"id={id_}, input={raw_text}\n "
                            f"Error: starts targets empty"
                        )
                        raise ValueError()
                    for st_r in starts_result:
                        search_hint["starts"].append(
                            [x.strip(" ") for x in st_r.split(",")]
                        )

                    if len(search_hint["search"]) != len(search_hint["starts"]):
                        if (
                            len(search_hint["search"]) > 1
                            and len(search_hint["starts"]) == 1
                            # and all(
                            #     set(search).issuperset(search_hint["starts"][0])
                            #     for search in search_hint["search"]
                            # )
                        ):
                            search_hint["starts"] = search_hint["starts"] * len(
                                search_hint["search"]
                            )
                        elif (
                            len(search_hint["search"]) == 1
                            and len(search_hint["starts"]) > 1
                        ):
                            search_hint["starts"] = search_hint["starts"][
                                : len(search_hint["search"])
                            ]
                        else:
                            logging.info(
                                f"id={id_}, input={raw_text}\n "
                                f"Error: length of search and starts targets don't match"
                            )
                            raise ValueError()
                except ValueError:
                    failed_to_parse_num += 1
                    continue
                search_hints.append(search_hint)
            parsed_data[id_] = search_hints
        logging.info(
            f"{failed_to_parse_num}/{len(parsed_data)} parse failed, "
            f"{failed_to_parse_num/len(parsed_data)*100:.2f}%"
        )
        return parsed_data

    def get_user_confirmation(self):
        estimated_token_num = self.FORMATTED_PROMPT_EXPECTED_TOKENS * (
            len(self.dataset.train_data)
            + len(self.dataset.validate_data)
            + len(self.dataset.test_data)
        )
        estimated_price = (
            estimated_token_num * self.MODEL_PRICE_PER_THOUSAND_TOKENS / 1000
        )
        response = input(f"Estimated price: ${estimated_price:.2f}, Continue? (Y/N).")
        while response.lower() not in ("y", "n"):
            input("Please input Y or N.")
        if response.lower() == "y":
            return
        else:
            print("Canceled, won't continue.")
            sys.exit(0)

    def get_full_prompt(self, base_prompt, query):
        return base_prompt.format(query=query)

    def generator(self, id_):
        data = self.generation_table[id_]
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY not set in environment")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            model=self.MODEL,
            prompt=self.get_full_prompt(self.BASE_PROMPT, data["text_question"],),
            max_tokens=128,
            temperature=0.7,
            stop=["Query:"],
            n=1,
        )
        return response["choices"]

    def __reduce__(self):
        return CommonsenseQA2Prompter, (self.dataset,)
