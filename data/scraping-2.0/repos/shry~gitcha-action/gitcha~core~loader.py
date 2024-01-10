"""
Custom gitcha loader
"""
import logging
import os
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader

from .schemas import GitchaYaml
from .utils import normalize_path

logger = logging.getLogger(__name__)


def _is_visible(path: Path) -> bool:
    """copy from DirectoryLoader
    """
    parts = path.parts
    for _p in parts:
        if _p.startswith('.'):
            return False
    return True


class GitchaDirectoryLoader(DirectoryLoader):
    """
    Custom gitcha loader function
    """

    def load(self, **kwargs) -> List[Document]:
        """Load documents."""

        gitcha = kwargs.get('gitcha')
        if not gitcha or not isinstance(gitcha, GitchaYaml):
            raise ValueError('No gitcha config provided')

        docs = []

        gitcha_config = gitcha.config.dict() if gitcha.config else {}

        items = None
        for entity in ['README.md', 'public', 'certificats', 'work_history', 'projects']:

            if entity != 'README.md':
                folder_name = gitcha_config.get(f'{entity}_folder')
                if not folder_name:
                    continue

                sub_path = normalize_path(self.path, folder_name)

                items = sub_path.rglob(
                    self.glob) if self.recursive else sub_path.glob(self.glob)

            else:
                # The root readme.md file will also be checked
                sub_path = Path(self.path)
                items = [Path(os.path.join(self.path, entity)), ]

            for i in items:
                if i.is_file():

                    # We currently do not support images
                    if str(i).lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif')):
                        continue

                    if _is_visible(i.relative_to(sub_path)) or self.load_hidden:
                        try:
                            sub_docs = self.loader_cls(
                                str(i), **self.loader_kwargs).load()
                            docs.extend(sub_docs)
                        except Exception as exc:  # pylint: disable=broad-exception-caught
                            if self.silent_errors:
                                logger.warning(exc)
                            else:
                                raise exc
        return docs
