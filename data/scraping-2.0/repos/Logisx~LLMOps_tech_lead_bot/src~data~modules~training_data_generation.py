import openai
from tqdm import tqdm
from typing import Dict

from src.utils.logging import logger
from config.paths import DATA_EXTERNAL_DIR, DATA_PROCESSED_DIR
from src.utils.file_management import FileManagement
from src.utils.configuration_management import ConfigurationManagement


class TrainingDataGenerator:
    def __init__(self):
        openai.api_key = ConfigurationManagement.get_openai_api_key()
        self.__params = ConfigurationManagement.get_training_data_generation_params()
        self.__examples = FileManagement.read_json(DATA_EXTERNAL_DIR / self.__params.examples_filename)

    def run(self):
        output = []
        for example in tqdm(self.__examples):
            prompt = self.__build_prompt(example)
            logger.info(f"{prompt=}")

            response = openai.Completion.create(
                engine=self.__params.engine,
                prompt=prompt,
                temperature=self.__params.temperature,
                max_tokens=self.__params.max_tokens,
            )

            response = response["choices"][0]["text"]
            logger.info(f"{response=}")

            output.append({**example, "response": response})

        logger.info("Saving training data to json")
        FileManagement.save_to_json(DATA_PROCESSED_DIR / "training_data.json", output)

    def __build_prompt(self, example: Dict) -> str:
        return self.__params.prompt_template.format(
            ABOUT_ME=example["about_me"],
            CONTEXT=example["context"],
        )