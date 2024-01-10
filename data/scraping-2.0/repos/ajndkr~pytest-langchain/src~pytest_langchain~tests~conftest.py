import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from langchain.chains import load_chain

ALLOWED_EXTENSIONS = [".yaml"]
PYTEST_CLI_ERROR = 4
PYTEST_INTERNAL_ERROR = 3


def pytest_generate_tests(metafunc):
    if "test_case" in metafunc.fixturenames:
        config = load_config(
            metafunc.config.getoption("--langchain-config-file"),
            metafunc.config.getoption("--langchain-config-dir"),
        )
        metafunc.parametrize("test_case", config["test_cases"])


@pytest.fixture(scope="session")
def config_file(request):
    return request.config.getoption("--langchain-config-file")


@pytest.fixture(scope="session")
def config_dir(request):
    return request.config.getoption("--langchain-config-dir")


@pytest.fixture(scope="session")
def config(config_file, config_dir):
    return load_config(config_file, config_dir)


@pytest.fixture(scope="session")
def chain_file(config: Dict[str, Any]):
    return config["chain_file"]


@pytest.fixture(scope="session")
def llm_chain(request, chain_file, config_dir):
    """Loads and tests LLMChain"""
    if not Path(chain_file).exists():
        chain_file = str(Path(config_dir) / Path(chain_file))

    os.environ["OPENAI_API_KEY"] = request.config.getoption("--openai-api-key")

    print("OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])

    return load_chain(chain_file)


def load_config(file: str, dir: str) -> Dict[str, Any]:
    """Loads config dictionary from path"""
    if file is None:
        pytest.exit("Config file not provided", PYTEST_CLI_ERROR)

    if not Path(file).exists():
        file = str(Path(dir) / Path(file))

    if Path(file).suffix not in ALLOWED_EXTENSIONS:
        pytest.exit("Invalid config file extension", PYTEST_CLI_ERROR)

    config = yaml.safe_load(Path(file).read_text())
    if "test_cases" not in config:
        pytest.exit(
            "Config file does not contain a 'test_cases' key", PYTEST_INTERNAL_ERROR
        )

    return config
