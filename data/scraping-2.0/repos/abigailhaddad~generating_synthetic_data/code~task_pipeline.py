import logging
from openai_agent import OpenAIAgent
from file_manager import FileManager
from data_classes import BaseTask, GeneratedText
from typing import List
from marvin import settings
from prompts_for_model import list_of_prompts

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class TaskGenerationPipeline:
    def __init__(self, base_task_str: str, prompts: List[str]):
        self.openai_agent = OpenAIAgent()
        self.base_task = BaseTask(base_task_str)
        self.prompts = prompts
        self.generated_texts = []

        logging.debug(f"Initialized TaskGenerationPipeline with base_task: {base_task_str}")

    def run(self, use_existing_files=False, base_task_runs=1, other_task_runs=1) -> List[GeneratedText]:
        logging.debug(f"Running pipeline with base_task_runs: {base_task_runs}, other_task_runs: {other_task_runs}")
        if use_existing_files:
            self.load_data_from_files()
        else:
            self.generate_and_save_data(base_task_runs=base_task_runs, other_task_runs=other_task_runs)
        return self.generated_texts

    def generate_and_save_data(self, base_task_runs=1, other_task_runs=1):
        logging.info("Generating base task data...")
        base_task_data = self.base_task.to_dict()
        FileManager.save_to_json(base_task_data, '../results/base_task.json')

        logging.info("Generating texts for tasks...")
        self.generated_texts = self.generate_texts_for_task(self.base_task, self.prompts, base_task_runs=base_task_runs, other_task_runs=other_task_runs)

        logging.debug(f"Generated texts count: {len(self.generated_texts)}")

        all_generated_texts = [text.to_dict() for text in self.generated_texts]
        FileManager.save_to_json(all_generated_texts, '../results/generated_texts.json')

    def load_data_from_files(self):
        logging.info("Loading data from files...")
        self.base_task = BaseTask.from_dict(FileManager.load_from_json('../results/base_task.json'))
        loaded_texts = FileManager.load_from_json('../results/generated_texts.json')
        self.generated_texts = [GeneratedText.from_dict(data) for data in loaded_texts]

    def generate_texts_for_task(self, base_task: BaseTask, queries: List[str], model: str = None, base_task_runs: int = 1, other_task_runs: int = 1) -> List[GeneratedText]:
        logging.debug(f"Generating texts for task with base_task_runs: {base_task_runs}, other_task_runs: {other_task_runs}")

        if model is None:
            model = settings.llm_model

        # Select tasks with repetition
        all_tasks = [(base_task.task, 'base_task')] * base_task_runs \
              + [(task, 'different_tasks') for task in base_task.different_tasks] * other_task_runs \
              + [(task, 'similar_tasks') for task in base_task.similar_tasks] * other_task_runs \
              + [(task, 'others') for task in base_task.others] * other_task_runs

        logging.debug(f"Total tasks generated for processing: {len(all_tasks)}")

        # Generating responses for all combinations of queries and tasks, then storing them in GeneratedText instances
        generated_texts = []

        for query in queries:
            for task, category in all_tasks:
                prompt = f"{query} {task}"
                logging.debug(f"Generating response for prompt: {prompt}")
                response_text = self.openai_agent.call_openai(prompt, model)
                logging.debug(f"Received response: {response_text}")
                generated_texts.append(GeneratedText(response_text, prompt, query, category))

        return generated_texts

    @property
    def generated_texts(self):
        return self._generated_texts

    @generated_texts.setter
    def generated_texts(self, value):
        self._generated_texts = value

    def save_generated_texts(self):
        logging.info("Saving updated generated texts to file...")
        all_generated_texts = [text.to_dict() for text in self.generated_texts]
        FileManager.save_to_json(all_generated_texts, '../results/generated_texts.json')


if __name__ == "__main__":
    pipeline = TaskGenerationPipeline("How to cook pasta puttanesca", list_of_prompts)
    generated_texts_result = pipeline.run(use_existing_files=False)
    for text in generated_texts_result:
        print(text)
