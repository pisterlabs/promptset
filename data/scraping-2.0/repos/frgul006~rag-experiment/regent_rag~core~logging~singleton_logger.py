import logging
import os
from typing import Any, Dict, Optional

import langchain
from colorama import Fore, Style
from dotenv import load_dotenv

# Define logging levels
LEVELS: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    """
    This class extends the logging.Formatter class to add color to the logs based on their level.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        This method formats the log messages.
        """
        # Assign colors based on the log level
        if record.levelno == logging.CRITICAL:
            prefix = f"{Fore.RED}{Style.BRIGHT}"
        elif record.levelno == logging.ERROR:
            prefix = f"{Fore.RED}"
        elif record.levelno == logging.WARNING:
            prefix = f"{Fore.YELLOW}"
        elif record.levelno == logging.INFO:
            prefix = f"{Fore.WHITE}"
        else:  # DEBUG and anything else
            prefix = f"{Fore.LIGHTBLACK_EX}"
        message = super().format(record)
        message = f"{prefix}{message}{Style.RESET_ALL}"
        return message


class SingletonLogger:
    """
    This class implements the Singleton design pattern for a logger.
    """

    _instance: Optional["SingletonLogger"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "SingletonLogger":
        """
        This method ensures that only one instance of the logger is created.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        This method initializes the logger.
        """
        load_dotenv()
        log_level = LEVELS.get(os.environ.get("LOG_LEVEL", "INFO"), logging.INFO)
        self._logger = logging.getLogger("SingletonLogger")
        self._logger.setLevel(log_level)

        if log_level == logging.DEBUG:
            logging.basicConfig()
            langchain.debug = True

            # Set the log level for the default logger
            logging.getLogger().setLevel(log_level)

            # Set the log level for the requests logger
            requests_logger = logging.getLogger("http.client")
            requests_logger.setLevel(log_level)

        formatter = ColoredFormatter("%(asctime)s %(levelname)s %(message)s")

        # StreamHandler for console logging
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        # FileHandler for file logging if LOG_FILE is set
        log_file = os.environ.get("LOG_FILE")
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self._logger.addHandler(fh)

        # Handler for uncaught exceptions
        logging.captureWarnings(True)

    def critical(self, *args: Any, **kwargs: Any) -> None:
        """
        This method logs a message with CRITICAL level.
        """
        self._logger.critical(*args, **kwargs)

    def error(self, *args: Any, **kwargs: Any) -> None:
        """
        This method logs a message with ERROR level.
        """
        self._logger.error(*args, **kwargs)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        """
        This method logs a message with WARNING level.
        """
        self._logger.warning(*args, **kwargs)

    def info(self, *args: Any, **kwargs: Any) -> None:
        """
        This method logs a message with INFO level.
        """
        self._logger.info(*args, **kwargs)

    def debug(self, *args: Any, **kwargs: Any) -> None:
        """
        This method logs a message with DEBUG level.
        """
        self._logger.debug(*args, **kwargs)
