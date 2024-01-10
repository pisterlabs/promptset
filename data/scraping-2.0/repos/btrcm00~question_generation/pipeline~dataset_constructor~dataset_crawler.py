import os
import openai

from tqdm import tqdm
from threading import Thread, Event
from queue import Queue

from common.config import PipelineConfig
from common.utils import *


class QGCrawler:
    def __init__(self, config: PipelineConfig = None) -> None:
        self.config = config if config is not None else PipelineConfig()
        openai.api_key = self.config.constructor_crawler_key

        self.qg = ModelUtils(input_max_length=self.config.pipeline_input_max_length)

        input_data_folder = f"{self.config.pipeline_dataset_folder}/source"
        assert os.path.isdir(input_data_folder)
        self.source_dataset = self.load_source_dataset(data_path=input_data_folder)

        output_folder = f"{self.config.pipeline_dataset_folder}/raw"
        check_exist_folder(output_folder)
        self.output_folder = output_folder
        self.output_queue = Queue(maxsize=10000)

    def prompt_sentence(self, pasage: str):
        return f"Generate 10 extractive questions in Vietnamese from the following passage (questions must be about information included in the passage, ask questions as specific as possible, and question is Wh-question). Passage: {pasage}"

    @property
    def prefix_data_id(self):
        return "chatgpt_data_"

    def make_request(self, content: str):
        completion = openai.ChatCompletion.create(
            model=self.config.constructor_openai_model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        question_text = completion.choices[0].message
        question_text = re.sub(r"\n+", "\n", question_text)
        question_text = re.sub(r"\d\.\s*", "", question_text)
        question_lst = question_text.split("\n")
        return question_lst

    def load_source_dataset(self, data_path: str = None):
        return json.load(open(data_path, "r", encoding="utf8"))

    def generate_qa_example(self, bar, q: Queue, e: Event):
        while not e.is_set() or not q.empty():
            data = q.get()
            base_id = self.prefix_data_id + data[0]
            question_lst = self.make_request(content=self.prompt_sentence(passage=data[1]))
            for q in question_lst:
                answer = self.qa_api(context=data[1], question=q)["data"][ANSWER].replace("_", " ")
                self.output_queue.put()
            bar.update(1)

    def write_output(self):
        pass

    def run(self):
        input_queue = Queue(maxsize=10000)
        bar = tqdm(total=len(self.qa_dataset), initial=0, leave=True)

        event = Event()
        event.clear()

        threads = [Thread(target=self.generate_qa_example, args=(bar, input_queue, event), daemon=True) for _ in
                   range(self.num_of_threads)]

        [thread.start() for thread in threads]

        for idx, ele in enumerate(self.source_dataset):
            input_queue.put((idx, ele))

        event.set()
        [thread.join() for thread in threads]

        for f in os.listdir(self.output_folder):
            if "all" in f:
                output += load_file(f"{self.output_folder}/{f}")
        self.save_dataset(output)


if __name__ == "__main__":
    pass
