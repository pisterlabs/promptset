"""CLI for the project."""
import os
import sys
import dotenv
import openai
from colorama import Fore, Style
from pyfiglet import Figlet
from .bots import ChatBotLoom
from .ui import run_tabbed_chat_ui

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    try:
        # load .env file from one directory up
        dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
        dotenv.load_dotenv(dotenv_path)
        bots_file_path = os.getenv("BOTS_FILE")
        loom = ChatBotLoom(bots_file_path)

        # Add some color and ASCII Art
        figlet = Figlet(font="digital")
        print(Fore.GREEN + figlet.renderText("Chat Bot Loom") + Style.RESET_ALL)

        # load bots from file if file exists
        if loom.chat_bots_file_exists() and len(loom.load_bots()) > 0:
            print(Fore.GREEN + f"Bots found in {bots_file_path}" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + f"No bots found at {bots_file_path}" + Style.RESET_ALL)
            print(Fore.YELLOW + "Creating new bots file..." + Style.RESET_ALL)
            # create sample bot for creating new bots
            loom.create_sample_bot_file()
            loom.save_bots_to_file(bots_file_path)

        openai.api_key = os.getenv("OPENAI_API_KEY")
        # chat with the bot
        run_tabbed_chat_ui(loom)
    except KeyboardInterrupt:
        print(Fore.RED + "\nExiting..." + Style.RESET_ALL)
        sys.exit(0)
    except Exception as unexpected_error:  # pylint: disable=broad-exception-caught
        print(Fore.RED + f"Error: {unexpected_error}" + Style.RESET_ALL)
        sys.exit(1)
