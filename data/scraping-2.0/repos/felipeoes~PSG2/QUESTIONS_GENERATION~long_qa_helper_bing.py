import re
import json
import pandas as pd
from tqdm import tqdm
from typing import Generator, Tuple
# from bing_qa_generator import BingQAGenerator
from bing_qa_generator_skype import BingQAGenerator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from threading import Thread
from multiprocessing import Queue
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

CHECKPOINTS_DIR = Path("checkpoints_bing")
CSV_DIR = Path("csv_bing")


class CheckpointSaver(Thread):
    """Thread class to save the checkpoint"""

    def __init__(self, qa_helper: "QAHelper"):
        super().__init__(daemon=True)
        self.qa_helper = qa_helper

    def run(self, *args, **kwargs):
        self.qa_helper.save_checkpoint(*args, **kwargs)


class DataframeSaver(Thread):
    """Thread class to save the dataframe"""

    def __init__(self, qa_helper: "QAHelper"):
        super().__init__(daemon=True)
        self.qa_helper = qa_helper

    def run(self, *args, **kwargs):
        self.qa_helper.save_dataframe(*args, **kwargs)


class ContextQueueFeeder(Thread):
    """Thread class to feed the queue with context"""

    def __init__(
        self,
        qa_helper: "QAHelper",
        context_queue: Queue,
        contents: "Generator[Tuple[int, str, bytes]]",
    ):
        super().__init__(daemon=True)
        self.qa_helper = qa_helper
        self.context_queue = context_queue
        self.contents = contents

    def run(self):
        total_read = 0
        for file_index, file_name, file_content, question, answer in self.contents:
            # add `start_monitoring` as attribute to the queue
            if not hasattr(self.context_queue, "start_monitoring"):
                print("Setting context_queue.start_monitoring to True")
                self.context_queue.start_monitoring = True
            try:
                # since reading from RAW_TEXT (plain text), we don't need to extract text from pdf
                # text = file_content.decode("utf-8")
                if file_index < self.qa_helper.last_file_index:
                    continue
                
                text = file_content

                self.context_queue.put((file_index, file_name, text, question, answer))

                self.qa_helper.count_filtered_texts += 1

                if total_read % 1000 == 0:
                    print(f"Total files read in context queue feeder: {total_read}")

                total_read += 1

            except Exception as e:
                print(e)
                continue
            except KeyboardInterrupt:
                print(f"KeyboardInterrupt | File index: {file_index}")
                break

        print("ContextQueueFeeder finished")


class QAHelper:
    """Question and Answer helper class. It will generate the question and answer for the given documents."""

    def __init__(
        self,
        g_drive_folder_name: str,
        csv_input_path: str,
        filtering_keywords: list,
        max_chunk_length: int = 4000,
        chunk_overlap: int = 1000,
        num_drivers: int = 1,
    ):
        self.folder_name = g_drive_folder_name
        self.csv_input_path = csv_input_path
        self.filtering_keywords = filtering_keywords
        self.num_drivers = num_drivers
        self.regex = re.compile(
            r"\b(?:%s)\b" % "|".join(filtering_keywords), re.IGNORECASE
        )
        self.max_chunk_length = max_chunk_length
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_length,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        self.last_file_name = ""
        self.last_file_index = 0
        self.count_filtered_texts = 0
        self.checkpoint_saver = CheckpointSaver(self)
        self.dataframe_saver = DataframeSaver(self)
        self.load_checkpoint(self.save_checkpoint(0, ""))
        self.context_queue = Queue()
        self.generator = BingQAGenerator(
            contents_queue=self.context_queue, num_drivers=self.num_drivers
        )
        self.context_queue_feeder = None

    def save_checkpoint(self, file_index: int, file_name: str):
        """Saves the current state of the QAHelper. Utilizes folder name to save the checkpoint."""
        # get the last part of the folder name. Ex: if folder name is `folder1/folder2`, then `folder2` will be returned.
        dest_folder_name = str(self.folder_name.as_posix()).split("/")[-1]
        checkpoint_path = (
            CHECKPOINTS_DIR
            / f"{dest_folder_name}_{self.max_chunk_length}_context_checkpoint.json"
        )

        # check if the checkpoint directory exists
        if not CHECKPOINTS_DIR.exists():
            CHECKPOINTS_DIR.mkdir()

        # check if checkpoint file exists
        if Path(checkpoint_path).exists():
            checkpoint = self.load_checkpoint(checkpoint_path)

            # check if last file index is greater than the current file index before overwriting
            if checkpoint["last_file_index"] > file_index:
                print(
                    f"Current file index is {file_index}. Last file index is {checkpoint['last_file_index']}. Not overwriting the checkpoint."
                )
                return checkpoint_path

        # save the last file name
        with open(checkpoint_path, "w") as f:
            json.dump(
                {
                    "last_file_name": file_name,
                    "last_file_index": file_index,
                    "count_filtered_texts": self.count_filtered_texts,
                },
                f,
            )

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """Loads the checkpoint from the given path"""

        # check if the checkpoint exists
        if not Path(checkpoint_path).exists():
            return

        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

            # set the last file name and index
            self.last_file_name = checkpoint["last_file_name"]
            self.last_file_index = checkpoint["last_file_index"]
            self.count_filtered_texts = checkpoint["count_filtered_texts"]

        return checkpoint

    def save_dataframe(self, dataframe: pd.DataFrame, csv_path: str):
        """Converts the dataframe to csv and return path, which is in the type `{folder_name}_{max_chunk_length}_context.csv`"""
        csv_string = dataframe.to_csv(csv_path, index=False)
        return csv_path

    def load_dataframe(self, csv_path: Path):
        """Loads the dataframe checkpoint if exists"""
        # check if the checkpoint exists
        columns = [
            "file_index",
            "file_name",
            "prompt",
            "question",
            "answer",
            "new_questions",
            "new_long_answers",
        ]
        if not csv_path.exists():
            return pd.DataFrame(
                columns=[
                    "file_index",
                    "file_name",
                    "prompt",
                    "question",
                    "answer",
                    "new_questions",
                    "new_long_answers",
                ]
            )
        # select only the columns we need
        return pd.read_csv(csv_path)[columns]

    def run(self):
        """Runs the QAHelper"""
        # create csv directory if it doesn't exist
        if not CSV_DIR.exists():
            CSV_DIR.mkdir(parents=True, exist_ok=True)

        # get the last part of the folder name. Ex: if folder name is `folder1/folder2`, then `folder2` will be returned.
        dest_folder_name = str(self.folder_name.as_posix()).split("/")[-1]
        csv_path = CSV_DIR / f"{dest_folder_name}_{self.max_chunk_length}_context.csv"

        df = self.load_dataframe(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")

        input_csv = pd.read_csv(self.csv_input_path)
        contents = (
            (index, row["file_name"], row["prompt"], row["question"], row["answer"])
            for index, row in input_csv.iterrows()
        )

        self.context_queue_feeder = ContextQueueFeeder(
            self, self.context_queue, contents
        )
        self.context_queue_feeder.start()

        SAVE_DATAFRAME_EVERY = 10

        # build dataframe with context, question and answer.
        pbar = None
        while (
            self.context_queue_feeder.is_alive()
            or not self.context_queue.empty()
            or len(self.generator.get_working_drivers()) > 0
        ):
            # initialize pbar when the context_queue_feeder ends, total is the number in self.context_queue
            if pbar is None and not self.context_queue_feeder.is_alive():
                pbar = tqdm(total=self.context_queue.qsize())

            try:
                qa_obj = self.generator.generate_qa()
                if qa_obj is None:
                    continue

                file_index = qa_obj["file_index"]
                file_name = qa_obj["file_name"]
                prompt = qa_obj["prompt"]
                question = qa_obj["question"]
                answer = qa_obj["answer"]
                new_questions = qa_obj["new_questions"]
                new_long_answers = qa_obj["new_long_answers"]

                if prompt is None or len(new_long_answers) == 0 or len( 
                    new_questions) == 0:
                    print(
                        f"Skipping file {file_name} because of empty context, questions or answers"
                    )
                    if pbar:
                        pbar.update(1)
                    continue

                row = pd.DataFrame(
                    [
                        {
                            "file_index": file_index,
                            "file_name": file_name,
                            "prompt": prompt,
                            "question": question,
                            "answer": answer,
                            "new_questions": new_questions,
                            "new_long_answers": new_long_answers,
                        }
                    ]
                )

                df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)

                if len(df) % SAVE_DATAFRAME_EVERY == 0:
                    print(f"Saving dataframe with {len(df)} rows")
                    self.dataframe_saver.run(df, csv_path)

                self.checkpoint_saver.run(file_index, file_name)

                if pbar:
                    pbar.update(1)

            except Exception as e:
                print(e)
                continue
            except KeyboardInterrupt:
                print(
                    f"KeyboardInterrupt | Dataframe length: {len(df)} | Csv path: {csv_path}"
                )
                break

        return df, csv_path
