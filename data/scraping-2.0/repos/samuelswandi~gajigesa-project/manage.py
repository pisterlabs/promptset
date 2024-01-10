#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import openai
from dotenv import load_dotenv


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    load_dotenv()  # take environment variables from .env
    openai.api_key = os.getenv('OPENAI_API_KEY') # load api key from .env

    main()
