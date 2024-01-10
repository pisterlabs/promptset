import os

import openai
import yaml
from util import logger

import pygpt

# Reading YAML file
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    globals().update(config)

root_folder = config["root_folder"]


class RepairRunner:
    repaires = []

    def __init__(self):
        # Initialize the class
        self.repairers = [OpenAIRepair()]

    def repair(self, code, analysis_results, test_results, debug_results):
        # Perform repairs on the code based on the analysis and test results
        # Return the repaired code
        results = []
        for repairer in self.repairers:
            result = repairer.repair(
                code, analysis_results, test_results, debug_results
            )
            results.append(result)
        return results


class Repair:
    def __init__(self):
        # Initialize the class
        return

    def repair(self, code, analysis_results, test_results, debug_results):
        # Perform repairs on the code based on the analysis and test results
        # Return the repaired code
        return


class OpenAIRepair(Repair):
    def __init__(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def repair(self, code, analysis_results, test_results, debug_results):
        # Generate the prompt
        prompt = f"Given the following code:\n{code}\n\nBased on the analysis results:\n{analysis_results}\n\nAnd the test results:\n{test_results}\n\nAnd the debug information:\n{debug_results}\n\nPlease avoid throw exception for the error\n\nPlease suggest a repaired version of the code:"
        logger.info("-------repair prompt---------")
        logger.info(prompt)
        # Generate a response using GPT-3
        response = pygpt.create(
            engine=config["model_engine1"],
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
        )

        # Extract the repaired code from the response
        repaired_code = response.choices[0].text.strip()
        logger.info("----------repaired code--------------")
        logger.info(repaired_code)
        # Return the repaired code
        return repaired_code
