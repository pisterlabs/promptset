from typing import List
import random
from typing import Dict, Any, Tuple
from openai_cli.completion.api import complete_prompt
import cv2
import skimage
from abcli.modules.host import is_jupyter
from abcli.modules import objects
from abcli import file
import numpy as np
from abcli.logging import crash_report
import abcli.logging
import logging

logger = logging.getLogger()


class ai_function(object):
    def __init__(
        self,
        function_name: str = "",
        verbose=None,
    ):
        self.language = "unknown"
        self.verbose = is_jupyter() if verbose is None else verbose

        self.function_name = (
            function_name
            if function_name
            else "{}_{}".format(
                self.__class__.__name__,
                random.randint(10000000, 99999999),
            )
        )

        self.prompt: List[str] = []
        self.code: List[str] = []

        self.auto_save = False

    def generate(
        self,
        prompt,
    ) -> Tuple[bool, Dict[str, Any]]:
        self.prompt = prompt

        logger.info("prompt: {}".format(prompt))

        success, self.code, metadata = complete_prompt(
            self.prompt,
            verbose=self.verbose,
        )

        if success:
            logger.info("code: {}".format(self.code))

        return success, metadata

    def save(self, filename=""):
        if not filename:
            filename = objects.path_of(f"{self.__class__.__name__}_code.json")

        _, content = file.load_json(filename, civilized=True)

        content[self.function_name] = self.to_json()

        file.save_json(filename, content)

        logger.info("-> {}".format(filename))

    def to_json(self):
        return {
            "code": self.code,
            "function_name": self.function_name,
            "prompt": self.prompt,
        }
