import logging
import os
import pickle
import time
from enum import Enum
from typing import Any

import openai

from quantgpt.financial_tools.utils import home_path

logger = logging.getLogger(__name__)


class DataCache:
    def __init__(
        self,
        cache_file=None,
        initial_prompt_file=None,
        final_prompt_file=None,
        overwrite_cache=False,
    ):
        self.cache = (
            self.load_object("cache", cache_file) if cache_file else {}
        )
        self.initial_prompt = (
            self.load_object("prompts", initial_prompt_file)
            if initial_prompt_file
            else ""
        )
        self.final_prompt = (
            self.load_object("prompts", final_prompt_file)
            if final_prompt_file
            else ""
        )
        self.overwrite_cache = overwrite_cache

    def fetch_summary(self, prompt, max_retries=3):
        retries = 0
        response_summary = None
        success = False

        while retries < max_retries and not success:
            try:
                response_summary = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are Bloomberg GPT, a Large Language Model which specializes in understanding financial data.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                success = True
            except Exception as e:
                logger.error(f"Error while fetching summary: {e}")
                retries += 1
                time.sleep(5)  # Wait for 5 seconds before retrying

        if not success:
            raise Exception("Failed to fetch summary after multiple attempts")

        return [prompt, response_summary]

    def load_prompt(self, file_path):
        with open(file_path, "r") as file:
            return file.read().strip()

    def build_prompt(self, title, body):
        return f"{self.initial_prompt} Title:\n{title}\nBody:\n{body}\n {self.final_prompt}"

    def get_result(self, title, body):
        prompt = self.build_prompt(title, body[0:3_000])
        if title in self.cache and not self.overwrite_cache:
            return self.cache[title]
        else:
            openai.api_key = (
                "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"
            )
            logger.info("Fetching sentiment for title: " + title + "...")
            result = self.fetch_summary(prompt)
            self.cache[title] = result[1]
            self.save_object(
                "cache", "cache.pkl", self.cache
            )  # Save the cache out.
            return result[1]

    def categorize_result(self, result):
        category = (
            result["choices"][0]["message"]["content"]
            .split("\n")[-1]
            .strip()
            .replace(".", "")
            .replace(",", "")
        )
        if ":" in category:
            category = category.split(":")[-1]
        category = category.strip().replace(".", "").replace(",", "")
        if "extremely positive".upper() in category.upper():
            category = self.Category("EXTREMELY_POSITIVE")
        elif "very positive".upper() in category.upper():
            category = self.Category("VERY_POSITIVE")
        elif "positive".upper() in category.upper():
            category = self.Category("POSITIVE")
        elif "neutral".upper() in category.upper():
            category = self.Category("NEUTRAL")
        elif "negative".upper() in category.upper():
            category = self.Category("NEGATIVE")
        else:
            category = self.Category("N/A")
        return category

    def save_object(self, obj_dir: str, file_name: str, obj: Any):
        file_path = os.path.join(
            home_path(),
            "data",
            "openai",
            obj_dir,
            file_name,
        )

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    def load_object(self, obj_dir: str, file_name: str):
        file_path = os.path.join(
            home_path(),
            "data",
            "openai",
            obj_dir,
            file_name,
        )

        if ".pkl" in file_name:
            # Saving the data object to a file
            with open(file_path, "rb") as file:
                obj = pickle.load(file)
            return obj
        elif ".txt" in file_name:
            with open(file_path, "r") as file:
                return file.read().strip()
        else:
            raise ValueError("Filetype not supported.")

    class Category(Enum):
        EXTREMELY_POSITIVE = "EXTREMELY_POSITIVE"
        VERY_POSITIVE = "VERY_POSITIVE"
        POSITIVE = "POSITIVE"
        NEUTRAL = "NEUTRAL"
        NEGATIVE = "NEGATIVE"
        N_A = "N/A"

        def __str__(self):
            return self.value
