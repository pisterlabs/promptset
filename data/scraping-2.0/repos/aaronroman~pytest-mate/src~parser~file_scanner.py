import os
import re
from loguru import logger


class FileScanner:
    def __init__(self, input_folder: str, extension: str, exclude: list = None):
        """
        :param input_folder: The folder to scan
        :param extension: The extension of the files to scan for
        :param exclude: A list of folders to exclude from the scan
        """
        self.ignore = ["__init__.py"]
        self.input_folder = input_folder
        self.extension = extension
        self.exclude = exclude if exclude is not None else []

    def scan(self):
        return self._scan_folder(self.input_folder)

    # singleton method to clean the code
    @staticmethod
    def get_code_from_codeblock(response: str) -> str:
        """
        :param response: The response from OpenAI
        :return: The code from the response
        """
        code = ""
        for pattern in [r"```python(.*)```", r"```(.*)```"]:
            try:
                code = re.search(pattern, response, re.DOTALL).group(1)
                break
            except AttributeError:
                pass
        return code

    def _scan_folder(self, folder_path):
        result = []

        # check exist folder folder_path
        if not os.path.isdir(folder_path):
            logger.critical(f"Folder {folder_path} does not exist")
            return result

        for entry in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry)

            if os.path.isdir(entry_path) and entry not in self.exclude:
                result.extend(self._scan_folder(entry_path))
            elif os.path.isfile(entry_path):
                _, file_extension = os.path.splitext(entry_path)
                if file_extension == self.extension and entry not in self.ignore:
                    result.append(entry_path)

        return result
