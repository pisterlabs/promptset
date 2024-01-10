import logging

from langchain.globals import set_verbose
from langchain_core.tracers import ConsoleCallbackHandler


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,  # Set the log level (e.g., INFO, DEBUG, ERROR)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]  # Use a stream handler to print to stdout
    )

    set_log_level('asyncio', logging.WARN)
    set_log_level('numexpr.utils', logging.WARN)

    set_log_level('aitestdrive.controller', logging.DEBUG)

    set_verbose(True)  # does not work with LCEL, manual callbacks are required for now -- see `log_to_console`


def log_to_console():
    return {'callbacks': [ConsoleCallbackHandler()]}


def set_log_level(base, level):
    logging.getLogger(base).setLevel(level)
