# file_manager.py
import logging
import os
from datetime import datetime

from openai.types.chat import ChatCompletion
import config
import json


class FileManager:
    """
    Utility class to handle file operations, such as reading, writing,
    and managing transcription files.
    """

    @staticmethod
    def get_datetime_string() -> str:
        """ Get the current datetime as a formatted string. """
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
        return timestamp

    @staticmethod
    def write_file(file_path: str, content: str):
        """ Write content to a file. """
        with open(file_path, "w") as outfile:
            outfile.write(content)

    @staticmethod
    def read_file(file_path: str) -> str:
        """ Read content from a file. """
        with open(file_path, "r") as infile:
            return infile.read()
    
    @staticmethod
    def read_json(file_path: str) -> dict:
        """ read json file and return contents as dict """
        data = None
        with open(file_path, "r") as infile:
            data = json.load(infile)
        return data

    @staticmethod
    def delete_transcription(file_name: str):
        """ Delete a transcription file. """
        file_path = os.path.join(config.TRANSCRIPTION_PATH, file_name)
        os.remove(file_path)

    @staticmethod
    def save_json(file_name: str, content: dict or list or ChatCompletion):
        """ Save content as a JSON file. """
        with open(file_name, "w") as f:
            json.dump(content, fp=f, indent=4)

    @staticmethod
    def mark_as_read(file_name: str):
        """ Mark a transcription file as read by renaming it. """
        original_file_path = os.path.join(config.TRANSCRIPTION_PATH, file_name)
        marked_file_path = os.path.join(config.TRANSCRIPTION_PATH, f"_read_{file_name}")
        os.rename(original_file_path, marked_file_path)

    @staticmethod
    def read_transcriptions(directory: str) -> list:
        """ Read transcriptions from a directory. """
        transcriptions = []
        for file in os.listdir(directory):
            if not file.startswith("_read_"):
                file_path = os.path.join(directory, file)
                if os.path.exists(file_path):
                    if os.stat(file_path).st_size != 0:
                        with open(file_path, "r") as f:
                            try:
                                transcription = json.load(f)
                                transcriptions.append(transcription)
                            except ValueError as e:
                                logging.error(f"Error reading {file_path}: {e}")
                        FileManager.mark_as_read(file)
                    else:
                        logging.error(f"File {file_path} is empty.")
                else:
                    logging.error(f"File {file_path} does not exist.")
            else:
                continue
        return transcriptions
