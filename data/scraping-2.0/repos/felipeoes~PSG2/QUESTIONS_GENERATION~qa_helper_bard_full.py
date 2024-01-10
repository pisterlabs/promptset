import fitz
import re
import json
import pandas as pd

from tqdm import tqdm
from typing import Generator, Tuple
from bard_qa_generator import BardQAGenerator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from threading import Thread
from multiprocessing import Queue
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from g_drive_service import (
    get_files_in_folder,
    read_folder_files_content,
    create_folder_or_get_id,
    upload_file,
    update_file,
    find_folder_id_by_name,
)

MIN_TEXT_LENGTH = 50

DATASETS_FOLDER_ID = "1wSDRHhI7omCCV5SrKeRgYrpbUFOD20Im"  # points to bard dataset
DATA_FOLDER_ID = "1vRRmyecRE71qKmHlmSbW29G1cPDj3E2t"  # where documents are stored
CHECKPOINTS_DIR = Path("checkpoints_bard")
CSV_DIR = Path("csv_bard")
RAW_TEXT_FOLDER = "RAW_TEXT"


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
        for file_index, file_name, file_content in self.contents:
            # add `start_monitoring` as attribute to the queue
            if not hasattr(self.context_queue, "start_monitoring"):
                print("Setting context_queue.start_monitoring to True")
                self.context_queue.start_monitoring = True
            try:
                # since reading from RAW_TEXT (plain text), we don't need to extract text from pdf
                text = file_content.decode("utf-8")
                if not self.qa_helper.check_text(text):
                    continue

                for part in self.qa_helper.text_splitter.split_text(text):
                    if self.qa_helper.check_text(part):
                        self.context_queue.put((file_index, file_name, part))

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
        filtering_keywords: list,
        service,  # gdrive service
        max_chunk_length: int = 4000,
        chunk_overlap: int = 1000,
        model_kwargs: dict = {},
        pipeline_kwargs: dict = {},
    ):
        self.folder_name = g_drive_folder_name
        self.filtering_keywords = filtering_keywords
        self.regex = re.compile(
            r"\b(?:%s)\b" % "|".join(filtering_keywords), re.IGNORECASE
        )
        self.service = service
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
        self.generator = BardQAGenerator(
            model_kwargs, contents_queue=self.context_queue
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

    def extract_text(self, file_path: Path = None, file_content: BytesIO = None):
        if file_path:
            doc = fitz.open(file_path)
        elif file_content:
            doc = fitz.open(stream=file_content, filetype="pdf")
        else:
            raise Exception("You must provide either a file path or a file content.")

        return "".join([page.get_text() for page in doc])

    def check_text(self, text: str):
        """Checks if text contains any of the filtering keywords"""
        return self.regex.search(text)

    def save_dataframe(self, dataframe: pd.DataFrame, csv_path: str):
        """Converts the dataframe to csv and return path, which is in the type `{folder_name}_{max_chunk_length}_context.csv`"""
        csv_string = dataframe.to_csv(csv_path, index=False)
        return csv_path

    def upload_dataframe(self, csv_path: Path):
        """Uploads the dataframe to GDrive. Destionation path is the `folder_name` inside `DATASETS_FOLDER_ID`. Create the folder if it doesn't exist."""
        # create folder and subfolders if it doesn't exist

        # self.folder_name can be of type `folder1/folder2/folder3`. So, we need to create all the subfolders. if not exist. And upload the file to the last subfolder.
        folders = str(self.folder_name.as_posix()).split("/")
        parent_folder_id = DATASETS_FOLDER_ID
        for folder in folders:
            parent_folder_id = create_folder_or_get_id(
                self.service, folder, parent_folder_id
            )

        # check if file already exist and update it
        file_id, file_parents, file_size = find_folder_id_by_name(
            self.service, csv_path.name, parent_folder_id
        )

        if file_id is not None:
            update_file(self.service, file_id, str(csv_path), mimetype="text/csv")
            return

        # upload file to the last subfolder
        upload_file(self.service, str(csv_path), parent_folder_id, mimetype="text/csv")

    def load_dataframe(self, csv_path: Path):
        """Loads the dataframe checkpoint if exists"""
        # check if the checkpoint exists
        columns = [ "file_index", "file_name", "prompt", "question", "answer" ]
        if not csv_path.exists():
            return pd.DataFrame(
                columns=[
                    "file_index",
                    "file_name",
                    # "context",
                    "prompt",
                    "question",
                    "answer",
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

        # get file_id of `folder_name`. It must be a subfolder like `icmbio/instrucoes_normativas`, So need to get the id of `instrucoes_normativas`
        # folder_id = DATA_FOLDER_ID
        # for folder in str(self.folder_name).split("/"):
        #     folder_id = find_folder_id_by_name(
        #         self.service, folder, parent_folder_id=folder_id
        #     )[0]

        # read txt files from `RAW_TEXT_FOLDER`, inside `folder_name`
        # files = get_files_in_folder(
        #     self.service,
        #     folder_name=RAW_TEXT_FOLDER,
        #     parent_folder_id=folder_id,
        #     pdf_only=False,
        # )

        # contents = read_folder_files_content(self, files)

        # get file names of files under `folder_name`. Local folder
        files = list(self.folder_name.glob("**/*.txt"))
        #   for future in futures:
        #         # save checkpoint
        #         file_index, file_name, result = future

        #         # save checkpoint
        #         # qa_helper.checkpoint_saver.run(file_index, file_name)

        #         # pbar.update(1)
        #         yield file_index, file_name, result.result()

        # generator for contents

        def read_file(file: Path):
            with open(file, "rb") as f:
                return f.read()

        contents = (
            (i, file.name, read_file(file))
            for i, file in enumerate(files)
            if i > self.last_file_index
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
                context = qa_obj["context"]
                questions = qa_obj["questions"]
                answers = qa_obj["answers"]

                if context is None or len(questions) == 0 or len(answers) == 0:
                    print(
                        f"Skipping file {file_name} because of empty context, questions or answers"
                    )
                    if pbar:
                        pbar.update(1)
                    continue

                rows = []
                qa_ord_dict = OrderedDict.fromkeys(zip(questions, answers))
                for  question, answer in qa_ord_dict:
                # for question, answer in zip(questions, answers):
                    # check if question and answer already exist
                    found = df.loc[
                        (df["question"] == question) & (df["answer"] == answer)
                    ]

                    if len(found) > 0:
                        continue
                    
                    rows.append(
                        {
                            "file_index": file_index,
                            "file_name": file_name,
                            "context": context,
                            "prompt": prompt,
                            "question": question,
                            "answer": answer,
                        }
                    )

                    # row = pd.DataFrame(
                    #     [
                    #         {
                    #             "file_index": file_index,
                    #             "file_name": file_name,
                    #             "context": context,
                    #             'prompt': prompt,
                    #             "question": question,
                    #             "answer": answer,
                    #         }
                    #     ]
                    # )

                    # df = pd.concat([df, row], ignore_index=True)

                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                
                if len(df) % SAVE_DATAFRAME_EVERY == 0:
                    print(f"Saving dataframe with {len(df)} rows")
                    # self.save_dataframe(df, csv_path)
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
