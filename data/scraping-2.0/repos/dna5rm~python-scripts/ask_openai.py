#!/bin/env -S python3
"""
OpenAI query & response script.

Example ~/.netrc file:
machine openai login api_key password sk-FyXXX...
"""

import logging
import netrc
import os
import sys
import textwrap
import openai
import re
import json
import curses
from openai.api_resources import model

# Function to display a supplied list
def curses_list(stdscr):
    '''Display a list in Curses'''

    # Clear the screen
    stdscr.clear()

    # Initialize the cursor position and the selection
    cursor_x = 0
    cursor_y = 0
    selected_item = 0

    # Set up the colors
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)

    stdscr.addstr(ctitle,0)

    # Loop until the user presses 'q'
    while True:
        # Print the citems, highlighting the selected item
        for index, item in enumerate(citems):
            if index == selected_item:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(index+2, 1, item)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(index+2, 1, item)

        # Refresh the screen
        stdscr.refresh()

        # Get the user's input
        c = stdscr.getch()

        # If the user pressed 'q', exit the loop
        if c == ord('q'):
            break
        # If user selected item
        elif c == ord('\n'):
            # return the selected item
            return citems[selected_item]
        # If the user pressed 'up', move the cursor up
        elif c == curses.KEY_UP:
            cursor_y = max(0, cursor_y - 1)
            selected_item = cursor_y
        # If the user pressed 'down', move the cursor down
        elif c == curses.KEY_DOWN:
            cursor_y = min(len(citems) - 1, cursor_y + 1)
            selected_item = cursor_y

# Function to get netrc credentials
def get_netrc_credentials(machine):
    """Fetch netrc credentials."""

    # Read in the netrc file
    netrc_file = netrc.netrc()

    try:
        machine_details = netrc_file.hosts[machine]
        return machine_details[0], machine_details[2]
    except KeyError:
        return None, None

# Function to ask OpenAI a question
def get_openai_text(task, **kwargs):
    """ OpenAI query for task. """

    # keywords & vars
    model = kwargs.get('model', 'code-davinci-003')

    # Get OpenAI credentials
    openai.api_key = get_netrc_credentials("openai")[1]

    if openai.api_key is None:
        print("No OpenAI credentials found.")
        sys.exit(1)

    # Get OpenAI response
    else:
        logging.info("OpenAi task: %s", task)
        response = openai.Completion.create(
            model=model,
            prompt=task,
            temperature=0.7,
            max_tokens=1900,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0)

    return response.choices[0].text

# function that will get a list of available openai model
def get_openai_models():
    """ Get list of available OpenAI models. """

    # Get OpenAI credentials
    openai.api_key = get_netrc_credentials("openai")[1]

    if openai.api_key is None:
        print("No OpenAI credentials found.")
        sys.exit(1)

    # Get OpenAI response
    else:
        models = openai.Engine.list()
        return models

if __name__ == "__main__":
    # Set environment basename for output files
    basename = os.path.splitext(os.path.basename(__file__))[0]

    # Read task from any type of stdin
    if not sys.stdin.isatty():
        message = sys.stdin.readlines()
    else:
        message = sys.argv[1:]

    # Initialize logging
    logfile = basename + '.log'
    logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)
    logging.info('-' * 80)

    # Get OpenAI response
    if (openai.api_key != 'None') and (message != []):
        ctitle = "Select OpenAI text model to use..."

        # Extract the IDs that match the specified pattern
        pattern = re.compile(r"^text-([a-z]+)-[0-9]+$")

        # Query OpenAI for models
        citems = [model["id"] for model in get_openai_models().data if pattern.match(model["id"])]
        citems.sort()

        # Display list for user selection
        model = curses.wrapper(curses_list)

        # Query OpenAI API for text
        if model != None:
            text = get_openai_text(message, model=model)
            logging.info(text)
            print(f"\033[44m\033[1mModel: {model}\033[0m")
            print(text)
        else:
            print("No GPT3 model selected...")

    else:
        print(basename + textwrap.dedent("""
        No query string to send to OpenAi...
        Example:
        $ ask_openai.py "Write ansible task to Ensure HTTP server is not enabled using REL 8 CIS benckmark"
        """))
        sys.exit(1)
