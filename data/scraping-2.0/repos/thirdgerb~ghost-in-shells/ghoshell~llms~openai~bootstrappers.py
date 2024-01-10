import logging
from typing import Dict

import yaml

from ghoshell.framework.ghost import GhostBootstrapper
from ghoshell.ghost import Ghost
from ghoshell.llms.openai.adapters import OpenAIConfig, OpenAIAdapter, OpenAIRecordStorage


class MockRecordStorage(OpenAIRecordStorage):

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def record(self, request: Dict, response: Dict | None, err: Exception | None) -> None:
        data = {
            "req >>>": request,
            "resp >>>": response,
        }
        self.logger.info(yaml.dump(data, allow_unicode=True))


class OpenAIBootstrapper(GhostBootstrapper):

    def __init__(self, relative_config_file: str = "llms/openai_config.yaml", logger_name: str = "llm"):
        self.relative_config_file = relative_config_file
        self.logger = logging.getLogger(logger_name)

    def bootstrap(self, ghost: Ghost):
        filename = ghost.config_path.rstrip("/") + "/" + self.relative_config_file.lstrip("/")
        with open(filename) as f:
            data = yaml.safe_load(f)
            config = OpenAIConfig(**data)
        storage = self._record_storage()
        adapter = OpenAIAdapter(config, storage)
        container = ghost.container
        for contract in adapter.contracts():
            container.set(contract, adapter)

    def _record_storage(self) -> OpenAIRecordStorage:
        return MockRecordStorage(self.logger)
