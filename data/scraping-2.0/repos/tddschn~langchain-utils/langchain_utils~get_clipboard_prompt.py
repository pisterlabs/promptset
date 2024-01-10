#!/usr/bin/env python3
"""
Author : Xinyuan Chen <45612704+tddschn@users.noreply.github.com>
Date   : 2023-10-11
Purpose: Get prompts from clipboard content
"""

import argparse
from pathlib import Path
from langchain_utils.get_text_prompt import get_args


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Get prompts from clipboard content - just use get-text-prompt with option -C, this script doesn\'t do anything.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return parser.parse_args()


def main():
    """Make a jazz noise here"""

    args = get_args()


if __name__ == '__main__':
    main()
