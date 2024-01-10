import os
from multiprocessing import Value

import click
import openai

from sum_pai.loguru_config import setup_logging
from sum_pai.process.directory import process_directory


@click.command()
@click.argument("directory_path")
@click.option(
    "--log-level",
    type=str,
    default=os.getenv("LOG_LEVEL", "INFO"),
    help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option(
    "--openai-api-key",
    type=str,
    default=os.getenv("OPENAI_API_KEY", None),
    help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
def main(directory_path, log_level, openai_api_key):
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Please set it to your OpenAI API key or pass the "
            "value to the --openai-api-key option."
        )
    openai.api_key = openai_api_key
    setup_logging(log_level)
    from loguru import logger

    logger.info("SumPAI - v0.3.0")
    logger.info(f"Logging is configured for {log_level} level.")
    process_directory(directory_path)


if __name__ == "__main__":
    main()
