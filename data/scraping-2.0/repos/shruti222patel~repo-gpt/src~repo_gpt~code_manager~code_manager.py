import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..console import verbose_print
from ..openai_service import OpenAIService
from .code_dir_extractor import CodeDirectoryExtractor
from .code_processor import CodeProcessor

logger = logging.getLogger(__name__)


class CodeManager:
    def __init__(
        self,
        output_filepath: Path,
        root_directory: Path = None,
        openai_service: OpenAIService = None,
    ):
        self.root_directory = root_directory
        self.output_filepath = output_filepath
        self.openai_service = (
            openai_service if openai_service is not None else OpenAIService()
        )
        self.code_processor = CodeProcessor(self.root_directory, openai_service)

        self.code_df = self.load_code_dataframe()
        self.directory_extractor = CodeDirectoryExtractor(
            self.root_directory, self.output_filepath, self.code_df
        )

    def display_directory_structure(self):
        structured_output = []
        for current_path, directories, files in os.walk(self.root_directory):
            depth = current_path.replace(self.root_directory, "").count(os.sep)
            indent = "    " * (depth)
            structured_output.append(f"{indent}/{os.path.basename(current_path)}")
            sub_indent = "    " * (depth + 1)
            for file in sorted(files):
                structured_output.append(f"{sub_indent}{file}")

        return "\n".join(structured_output)

    def load_code_dataframe(self):
        dataframe = None
        if os.path.exists(self.output_filepath):
            with open(self.output_filepath, "rb") as file:
                loaded_data = pickle.load(file)
            dataframe = pd.DataFrame(loaded_data)
        return dataframe

    def setup(self):
        self._extract_process_and_save_code()

        logger.verbose_info("All done! ✨ 🦄 ✨")

    def _store_code_dataframe(self, dataframe):
        output_directory = Path(self.output_filepath).parent

        if not output_directory.exists():
            output_directory.mkdir(parents=True)
            print(f"Directory created: {output_directory}")

        # Save DataFrame as a pickle file
        with open(self.output_filepath, "wb") as file:
            pickle.dump(dataframe, file)

    def _extract_process_and_save_code(self):
        (
            extracted_code_blocks,
            outdated_checksums,
        ) = self.directory_extractor.extract_code_blocks_from_files()
        processed_dataframe = self.code_processor.process(extracted_code_blocks)

        updated_df = pd.concat([self.code_df, processed_dataframe], ignore_index=True)

        # Remove checksums of updated code
        updated_df = updated_df[~updated_df["file_checksum"].isin(outdated_checksums)]

        self._store_code_dataframe(updated_df)
