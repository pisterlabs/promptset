import argparse
from azext_copilot.conversation_engine import ConversationEngine
from azext_copilot.services.authentication import AuthenticationService
from azext_copilot.services.openai import OpenAIService
from azext_copilot.helpers import execute
from azext_copilot.configuration import check_config, read_config
from azext_copilot.constants import (
    AUTO_RUN_CONFIG_KEY,
    COMMAND_KEY,
    COPILOT_CONFIG_SECTION,
    ENABLE_LOGGING_CONFIG_KEY,
    PROBLEM_KEY,
    EXPLANATION_KEY,
    SHOW_COMMAND_CONFIG_KEY,
)


def invoke():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("autorun", type=bool, default=False, nargs="?")
    args = parser.parse_args()
    copilot(args.prompt, args.autorun)


def copilot(prompt, autorun=False):
    # Get configuration values
    config = read_config()

    # Determine if configuration has been set
    check_state = check_config(config)
    if check_state != "":
        print(
            f"Configuration was found with empty values. [{check_state}]."
            "\nUse 'az copilot config set' to set the config values. "
            "Run 'az copilot config set --help' to see options "
            "or 'az copilot config show' to see current values."
        )
        return

    # Set variables from config
    enable_logging = config[COPILOT_CONFIG_SECTION][ENABLE_LOGGING_CONFIG_KEY]
    should_autorun = autorun or config[COPILOT_CONFIG_SECTION][AUTO_RUN_CONFIG_KEY]
    show_command = config[COPILOT_CONFIG_SECTION][SHOW_COMMAND_CONFIG_KEY]

    # Check authentication
    authentication_service = AuthenticationService()

    # If the user is not authenticated, display a message and exit
    if not authentication_service.is_authenticated():
        print(
            "You are currently not authenticated. "
            "Login with 'az login' before continuing."
        )
        return

    # Setup OpenAI service
    openai = OpenAIService(config)

    # Setup conversation engine
    engine = ConversationEngine(openai, config)

    # Send the prompt to the engine
    response = engine.send_prompt(prompt)

    # Run the feedback loop
    while not engine.is_finished():
        print("\nSorry, I need some more information:")
        print(f"\nCommand: {response[COMMAND_KEY]}")
        if response[EXPLANATION_KEY] != "":
            print(f"Explanation: {response[EXPLANATION_KEY]}")
        print(f"Problem: {response[PROBLEM_KEY]}")
        prompt = input(f"\n{response[PROBLEM_KEY].rstrip('.')}: ")

        # Check if the user wants to quit
        if prompt == "quit" or prompt == "q":
            print("Quitting Copilot conversation.")
            return

        response = engine.send_prompt(prompt)

    # If autorun is True just execute the command
    if should_autorun is True:
        # Check if the user wants to see the command
        if show_command is True:
            print(f"\nCommand: {response[COMMAND_KEY]}")
            print(f"Explanation: {response[EXPLANATION_KEY]}")

        # Execute the command and show the output
        stdout, stderr, has_err = execute(response[COMMAND_KEY], enable_logging)

        # If an unhandled error occurred then drop out
        if has_err:
            print(stderr)
            exit(0)

        # Show the output of the command
        print(stdout)
    else:
        # Show Command setting is ignored here because you can't make
        # a choice if you don't see the command first
        print(f"\nCommand: {response[COMMAND_KEY]}")
        print(f"Explanation: {response[EXPLANATION_KEY]}")
        run_command = (
            input("\nDo you want to execute this command? (Y/n) ").lower().strip()
        )
        # If the user wants to execute the command, do so
        if run_command == "y" or run_command == "":
            # Execute the command and show the output
            stdout, stderr, has_err = execute(response[COMMAND_KEY], enable_logging)

            # If an unhandled error occurred then drop out
            if has_err:
                print(stderr)
                exit(0)

            # Show the output of the command
            print(stdout)
        else:
            print("Command not executed.")
