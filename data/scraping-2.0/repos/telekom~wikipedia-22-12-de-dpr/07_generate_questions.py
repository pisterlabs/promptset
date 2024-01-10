# Copyright (c) 2024 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

# python 07_generate_questions.py --arango-conf config_arangodb_question.env --openai-conf config_openai_azure.yaml --prompt-type question
# python 07_generate_questions.py --arango-conf config_arangodb_imperative.env --openai-conf config_openai_azure.yaml --prompt-type imperative

"""Generate questions for context with OpenAI LLMs."""

import random
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from functools import partial
from typing import Dict, Literal, Sequence

import backoff
from mltb2.arangodb import ArangoBatchDataManager
from mltb2.db import BatchDataProcessor
from mltb2.openai import OpenAiAzureChat, OpenAiChatResult

prompts: Dict[str, str] = {}  #

# the normal question prompt
prompts[
    "question"
] = """\
Create a list of 6 questions in German language. \
It must be possible to answer the questions based on the given text. \
The question must not contain the word "and" (German "und").

The given text in German language:

{context}

The list of 6 different questions in German language without the word "and" (German "und"):"""


# the imperative prompt
prompts[
    "imperative"
] = """\
Create a list of 6 short questions in imperative form. \
An imperative question is a type of question that is phrased as a command or an instruction. \
It must be possible to answer the imperative questions based on the given text. \
The imperative question must not contain the word "and" (German "und").

The given text in German language:

{context}

The list of 6 short questions in imperative form and German language:"""


def meta_data_factory():
    """Create meta data for the result."""
    return {
        "script_name": "07_generate_questions.py",
        "script_version": "1",
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def generate_normal_questions(
    batch: Sequence, prompt_type: Literal["question", "imperative"], open_ai_client
) -> Sequence:
    """Generate questions for context."""
    results = []
    _prompt = prompts[prompt_type]
    for doc in batch:
        llm_response: OpenAiChatResult = open_ai_client(
            prompt=_prompt.format(context=doc["context"]),
            completion_kwargs={"temperature": random.uniform(0.0, 1.0), "max_tokens": 750},  # noqa: S311
        )
        result = {
            "_key": doc["_key"],
            f"meta_data_{prompt_type}": meta_data_factory(),
            f"llm_response_{prompt_type}": asdict(llm_response),
        }
        results.append(result)
        print(result)
        print("---")
        print(llm_response.content)
        print()
    return results

@backoff.on_exception(backoff.constant, Exception, max_tries=100, interval=60, jitter=None)
def main() -> None:
    """Main function."""
    # read command line arguments
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--arango-conf", type=str, required=True)
    argument_parser.add_argument("--openai-conf", type=str, required=True)
    argument_parser.add_argument("--prompt-type", choices=["question", "imperative"])
    args = argument_parser.parse_args()

    # create openai client
    open_ai_azure_chat = OpenAiAzureChat.from_yaml(args.openai_conf)

    # add prompt_type and open_ai_client to generate_normal_questions as partial
    generate_normal_questions_partial = partial(
        generate_normal_questions, prompt_type=args.prompt_type, open_ai_client=open_ai_azure_chat
    )

    # create arango client
    arango_batch_data_manager = ArangoBatchDataManager.from_config_file(
        args.arango_conf,
        aql_overwrite="FOR doc IN @@coll FILTER !HAS(doc, @attribute) SORT RAND() LIMIT @batch_size RETURN doc",
    )
    batch_data_processor = BatchDataProcessor(
        data_manager=arango_batch_data_manager,
        process_batch_callback=generate_normal_questions_partial,
    )
    batch_data_processor.run()


if __name__ == "__main__":
    main()
