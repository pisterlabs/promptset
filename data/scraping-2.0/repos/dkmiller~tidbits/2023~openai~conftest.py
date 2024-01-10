import logging
import os
import time
from subprocess import Popen

import pytest
import requests

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def openai():
    import openai

    # https://github.com/openai/openai-python#microsoft-azure-endpoints
    openai.api_key = "sk-fake"
    openai.api_base = "http://localhost:8000"
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"

    base = os.environ["base"]
    api_key = os.environ["api_key"]

    process = None

    try:
        # https://github.com/dkmiller/tidbits/blob/master/2023/2023-03_build-k8s/src/ptah/core/process.py
        process = Popen(["mock-openai"])
        log.info("Spawned process %s", process.pid)
        response = None

        for _ in range(10):
            try:
                response = requests.post(
                    "http://localhost:8000/config",
                    json={"base": base, "api_key": api_key},
                )
                if response.ok:
                    log.info("ok")
                    break
            except:
                time.sleep(0.5)

        if not response or not response.ok:
            raise Exception("All attempts failed!")

        yield openai
    except BaseException:
        if process:
            process.kill()
        raise
