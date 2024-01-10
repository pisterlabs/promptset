import os
import openai
from pipeline.base import SynthPipeline
from pipeline.description import DescriptionBuilder
from pipeline.llama2 import Llama2SampleBuilder
from pipeline.sample import DataSampleBuilder
from pipeline.snippet import CodeSnippetBuilder
from pipeline.task import TaskBuilder
from dotenv import load_dotenv

load_dotenv()

import logging
import sys

# Configure root logger
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    pipeline = SynthPipeline(
        model="gpt-4-0613",
        temperature=1.0,
    )

    pipeline \
        .add_stage(Llama2SampleBuilder())
    #   .add_stage(DataSampleBuilder())

    # .add_stage(CodeSnippetBuilder()) \
    # .add_stage(DescriptionBuilder()) \
 
    # .add_stage(TaskBuilder()) \
    for id in [
        "airesearch",
        "business",
        "businesstraveller",
        "designconsultant",
        "freelancecoder",
        "personalassistant",
        "shopify",
        "twitch",
        "twitter",
        "uberdriver",
        "ubereats",
        "youtube",
    ]:
        pipeline.run(id)
