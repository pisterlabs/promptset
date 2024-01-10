"""
This module provides functionality to do a code review on Python code using the GPT-4 Turbo API

Classes:
    UTCFormatter

Functions:
    main
    review_file

Constants:
    OPENAI_API_KEY (str): The OpenAI API key loaded from .env file.
"""

ENV_FILE = ".env"

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytz
from dotenv import dotenv_values
from openai import OpenAI

from language_prompts import LANGUAGE_PROMPTS


class UTCFormatter(logging.Formatter):
    """Custom log formatter that converts times to UTC to maintain a global consistent timestamp."""

    def formatTime(self, record, datefmt=None):
        # Convert log record creation time to UTC
        record_utc = datetime.utcfromtimestamp(record.created).replace(tzinfo=pytz.utc)
        if datefmt:
            s = record_utc.strftime(datefmt)
        else:
            s = record_utc.isoformat(timespec="milliseconds")
        return s


class CodeReviewer:
    def __init__(self, client, language_prompts):
        self.client = client
        self.language_prompts = language_prompts

    def detect_language(self, file_path):
        ext = file_path.suffix.lower()
        if ext in self.language_prompts:
            return self.language_prompts[ext]
        raise ValueError("Unsupported file type")

    def review_file(self, file_path, model):
        try:
            logger = logging.getLogger(__name__)
            prompt = self.detect_language(file_path)
            with open(file_path, "r") as file:
                code_content = file.read()
                # Add line numbers to each line
                lines = code_content.splitlines()
                code_content = "\n".join(
                    f"{i + 1}: {line}" for i, line in enumerate(lines)
                )

            tag = self.get_tag()
            logger.info(f"tag = {tag}")

            # Modify here to use the correct API call as per your OpenAI library version
            response = self.client.chat.completions.create(
                model=model,
                # ... Other parameters ...
                messages=[prompt, self.construct_user_prompt(code_content, tag)],
                temperature=0,
            )

            self.log_review(response.choices[0].message.content)
        except ValueError as ve:
            logger.error(f"Value Error: {ve}")
            return False
        except OSError as ose:
            logger.error(f"OS Error: {ose}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return False
        return True

    @staticmethod
    def construct_user_prompt(code_content, tag):
        return {
            "role": "user",
            "content": f"I have the following code:\n\n```{tag}\n{code_content}\n```\n\nPlease conduct a thorough code review of the provided codebase. Focus on identifying critical improvements across the following areas, providing detailed explanations and, where possible, suggest actionable refactoring strategies:\n\n1. **Code Efficiency**: Pinpoint areas where the code could be optimized for performance. Look for common pitfalls such as unnecessarily complex algorithms, redundant code, or inefficient database queries.\n\n2. **Readability and Maintainability**: Assess the code for readability and maintainability. Recommend best practices such as consistent naming conventions, clear function and variable names, and the separation of concerns to make the code more understandable and easier to maintain.\n\n3. **Design Patterns and Architecture**: Evaluate the use of design patterns and overall architecture. Suggest refactoring opportunities that could make better use of object-oriented or functional programming principles to improve scalability and robustness.\n\n4. **Security Vulnerabilities**: Scrutinize the code for potential security vulnerabilities, such as exposure to SQL injection, cross-site scripting (XSS), or insecure data handling. Propose remedies adhering to secure coding standards.\n\n5. **Error Handling and Logging**: Highlight how the code handles exceptions and performs logging, recommending strategies to improve error reporting and resilience in the face of failures.\n\n6. **Testing and Coverage**: Comment on the current testing practices. Offer guidance on enhancing test coverage and incorporating different types of testing like unit, integration, and end-to-end tests, as appropriate.\n\n7. **Compliance with Best Practices and Standards**: Check for adherence to the language-specific and industry-wide best practices, including commenting, documentation, and code formatting standards.\n\n8. **Technical Debt Identification**: Identify any ‘quick fixes’ or temporary solutions that have resulted in technical debt, and suggest a prioritized plan for addressing these issues.\n\n9. **Dependencies Management**: Review any third-party libraries or dependencies for appropriateness, licensing issues, and ensure that they are up to date and maintained.\n\n10. **Code Patterns and Anti-Patterns**: Call out any anti-patterns and suggest alternative approaches or patterns that could enhance the overall quality and functionality of the code.\n\nBy targeting these specific areas, provide insights into optimizations, potential refactoring, improved practices, and preventative measures to enhance the quality, maintainability, and security of the provided codebase.\n\nPlease note that each line in the code starts with a line number then a colon and a space. Try to use the line number when referencing every code line or at least the start of the code block.",
        }

    def log_review(self, review_content):
        logger = logging.getLogger(__name__)
        logger.info(f"Review Content:\n{review_content}")


class PythonReviewer(CodeReviewer):
    def get_tag(self):
        return "python"


class TypeScriptReviewer(CodeReviewer):
    def get_tag(self):
        return "typescript"


class KotlinReviewer(CodeReviewer):
    def get_tag(self):
        return "kotlin"


class CppReviewer(CodeReviewer):
    def get_tag(self):
        return "cpp"


class ReviewerFactory:
    def __init__(self, client, language_prompts):
        self.client = client
        self.language_prompts = language_prompts
        self.reviewer_classes = {
            ".py": PythonReviewer,
            ".ts": TypeScriptReviewer,
            ".kt": KotlinReviewer,
            ".cpp": CppReviewer,
            # extend with more as needed...
        }

    def get_reviewer(self, file_extension):
        reviewer_class = self.reviewer_classes.get(file_extension)
        if reviewer_class is None:
            raise ValueError(f"Unsupported file type: {file_extension}")
        return reviewer_class(self.client, self.language_prompts)


def setup_logging(log_file):
    """Sets up logging to output both to console and to a file."""
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = UTCFormatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Create file handler and set level to debug
    file_handler = logging.FileHandler(log_file, mode="a")  # Append mode
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Example log message
    logger.info(f"{'=' * 60} APPLICATION STARTED")
    return logger


def initialize_configuration():
    """Initializes the application configuration from .env file and sets up logging."""
    if not os.path.exists(ENV_FILE):
        logging.error("Error: .env file not found.")
        sys.exit(1)

    config = dotenv_values(ENV_FILE)

    OPENAI_API_KEY = config.get("OPENAI_API_KEY")
    OPENAI_MODEL = config.get("OPENAI_MODEL", "gpt-4-1106-preview")
    LOG_FILE = config.get("LOG_FILE", "application.log")

    if not OPENAI_API_KEY:
        logging.error("Error: OPENAI_API_KEY not found in .env file.")
        sys.exit(1)

    if not OPENAI_MODEL:
        logging.error("Error: No model specified.")
        sys.exit(1)

    setup_logging(LOG_FILE)

    return OPENAI_API_KEY, OPENAI_MODEL, LOG_FILE


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Review Python code using the GPT-4 Turbo API"
    )
    parser.add_argument("--file", type=str, help="The input file path", required=True)
    return parser.parse_args()


def get_file_path(args):
    resolved_file_path = Path(args.file).resolve(strict=True)
    # logging.info(f"Found file '{resolved_file_path}'")
    return resolved_file_path


def create_client(api_key):
    return OpenAI(api_key=api_key)  # We omit the parameter if it's the default


def main():
    """Entry point of the application"""
    args = parse_arguments()
    try:
        OPENAI_API_KEY, OPENAI_MODEL, LOG_FILE = initialize_configuration()
        logger = logging.getLogger(__name__)
        resolved_file_path = get_file_path(args)

        if resolved_file_path.exists():
            logger.info(f"Found file '{resolved_file_path}'")

            ext = resolved_file_path.suffix.lower()
            client = create_client(OPENAI_API_KEY)
            factory = ReviewerFactory(client, LANGUAGE_PROMPTS)
            language_reviewer = factory.get_reviewer(ext)
            language_reviewer.review_file(resolved_file_path, OPENAI_MODEL)
        else:
            raise FileNotFoundError(
                f"The file '{resolved_file_path}' was not found. Please check the path and try again."
            )
    except FileNotFoundError as not_found_err:
        logger.error(f"File not found error: {not_found_err}")
        sys.exit(1)
    except ValueError as val_err:
        logger.error(f"Value error: {val_err}")
        sys.exit(1)
    except Exception as ex:
        logger.error(f"An unexpected error occurred: {ex}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
