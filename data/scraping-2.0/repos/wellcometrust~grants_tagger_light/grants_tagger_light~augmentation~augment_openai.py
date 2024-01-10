import datetime
import json
import math
import os
import uuid

import openai

from loguru import logger
import numpy as np

from grants_tagger_light.augmentation.JsonParser import JsonParser


class AugmentOpenAI:
    def __init__(self, prompt_template_path, model_key="gpt-3.5-turbo"):
        if "OPENAI_API_KEY" not in os.environ:
            logger.error(
                "OPENAI_API_KEY not found in env vars. "
                "Please define it before running this program."
            )
        with open(prompt_template_path, "r") as f:
            self.prompt_template = f.read()
        self.model_key = model_key

    def _create_message(self, abstract, tag):
        prompt = self.prompt_template.replace("{TOPIC}", tag)
        prompt = prompt.replace("{ABSTRACT}", abstract)

        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _parse_response(answer, metadata):
        print(json.dumps(answer, indent=2))
        with open(metadata["save_to_path"], "a") as f:
            try:
                json_response = JsonParser.parse_json(answer)
            except Exception as e:
                logger.info(f"Error processing output: {e}. Skipping...")
                return

            res = {
                "journal": metadata["model_key"],
                "meshMajor": metadata["tags"],
                "year": metadata["year"],
                "abstractText": json_response["abstract"]
                .replace("'", "")
                .replace('"', ""),
                "pmid": uuid.uuid4().hex,
                "title": json_response["title"].replace("'", "").replace('"', ""),
                "existing_example": metadata["existing_example"]
                .replace("'", "")
                .replace('"', ""),
                "required_examples": metadata["required_examples"],
                "featured_tag": metadata["featured_tag"],
            }

            f.write(json.dumps(res))
            f.write("\n")
            f.flush()

            logger.info(f"Data received successfully for {metadata['featured_tag']}")

    @staticmethod
    def process_choices(choices, metadata):
        for c in choices:
            if "message" in c:
                if "content" in c["message"]:
                    AugmentOpenAI._parse_response(c["message"]["content"], metadata)

    @staticmethod
    def _process_response(result):
        AugmentOpenAI.process_choices(result.choices, result.metadata)

    def _prepare_request(
        self,
        tag,
        missing_num,
        dset,
        temperature,
        top_p,
        presence_penalty,
        num_proc,
        model_key,
        save_to_path,
    ):
        year = datetime.date.today().year

        logger.info(f"Augmenting {tag} with {missing_num} examples")
        # RAG: I select similar articles to provide them to the LLM

        tmp_dset = dset.filter(
            lambda x: any(np.isin([tag], x["meshMajor"])), num_proc=num_proc
        )

        # abstracts_num = [i for i in range(len(tmp_dset))]
        # random.shuffle(abstracts_num)

        required_examples = missing_num
        existing_examples = min(required_examples, len(tmp_dset))

        n_per_example = math.ceil(required_examples / existing_examples)
        logger.info(
            f"Augmenting {tag} with {required_examples} examples, "
            f"using {existing_examples} in RAG mode"
        )

        for i in range(existing_examples):
            abstract = tmp_dset["abstractText"][i]
            tags = tmp_dset["meshMajor"][i]
            data = {
                "model": self.model_key,
                "n": n_per_example,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "messages": self._create_message(abstract, tag),
            }

            metadata = {
                "featured_tag": tag,
                "tags": tags,
                "required_examples": missing_num,
                "existing_example": abstract,
                "year": year,
                "model_key": model_key,
                "save_to_path": save_to_path,
            }

            yield data, metadata

    def _make_requests(
        self,
        collect_concurrent_calls,
        dset,
        temperature,
        top_p,
        presence_penalty,
        num_proc,
        model_key,
        save_to_path,
    ):
        for num in range(len(collect_concurrent_calls)):
            tag = collect_concurrent_calls[num][0]
            missing_num = collect_concurrent_calls[num][1]

            for data, metadata in self._prepare_request(
                tag,
                missing_num,
                dset,
                temperature,
                top_p,
                presence_penalty,
                num_proc,
                model_key,
                save_to_path,
            ):
                chat_completion = openai.ChatCompletion.create(**data)
                chat_completion.metadata = metadata

                self._process_response(chat_completion)

    def generate(
        self,
        collect_concurrent_calls,
        dset,
        save_to_path,
        model_key,
        temperature=1.5,
        top_p=1,
        presence_penalty=0,
        num_proc=os.cpu_count(),
    ):
        self._make_requests(
            collect_concurrent_calls=collect_concurrent_calls,
            dset=dset,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            num_proc=num_proc,
            model_key=model_key,
            save_to_path=save_to_path,
        )
