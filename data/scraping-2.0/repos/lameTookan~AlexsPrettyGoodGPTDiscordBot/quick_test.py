import sys
import time

RED = "\u001b[31m"
BLUE = "\u001b[34m"
GREEN = "\u001b[32m"
RESET = "\u001b[0m"
MAGENTA = "\u001b[35m"


sys.path.append("./APGCM")
results = []
def finish():
    print("Tests finished.")
    print("Results:")
    print_results()
    print("Thank you for using APGCM!")
    input("Press enter to exit....")
    sys.exit()
def skip_test(step_numb: int, *args) -> None:
    """Append a skipped message to the results list, print the message, and then wait for the user to press enter."""
    global results
    results.append(f"{BLUE} STEP {step_numb} SKIPPED....{RESET}")
    print(f"{BLUE} STEP {step_numb} SKIPPED....{RESET}")
    for arg in args:
        print(arg)
    input("Press enter to continue....")
def print_results():
    global results
    print("....(Test Results)....")
    for numb, result in enumerate(results):
        numb += 1
        print(f" Test {numb}....... {result}")
        


def passed(step_numb: int, *args) -> None:
    """Append a passed message to the results list, print the message, and then wait for the user to press enter."""
    global results
    results.append(f"{GREEN} STEP {step_numb} PASSED....{RESET}")
    print(f"{GREEN} STEP {step_numb} PASSED....{RESET}")
    for arg in args:
        print(arg)
    input("Press enter to continue....")


def failure(step_numb: int, fail_msg=None, *args) -> None:
    """Print a failure message and exit the program, appending the failure message to the results list, printing the results, and then exiting."""
    global results
    if fail_msg is None:
        fail_msg = "An unknown error has occurred."

    results.append(f"{RED} STEP {step_numb} FAILED....{RESET}")
    print(f"{RED} STEP {step_numb} FAILED....{RESET}")
    for arg in args:
        print(arg)
    print(f"Error message: {fail_msg}")
    print_results()
    input("Press enter to exit....")
    print("Exiting...")


"""A simple test script meant to ensure users have correctly setup the project, before running the full program.
4 steps:
1. Checking that you have installed at least some of the dependencies.
2. Printing out your current settings to ensure that's been setup correctly.
3. Getting a test response from the AI to ensure that you have correctly entered your API key and template name.
4. Trying to send a discord message to ensure that you have correctly setup the discord bot.
"""
divider1 = "===================================================================================================="
divider1_len = len(divider1)
divider2 = "---------------------" * 3


def print_header() -> None:
    header = [
        divider1,
        "\n",
        f"{MAGENTA} <============== QUICK SETUP TEST SCRIPT ==============>{RESET}".center(
            divider1_len
        ),
        " ",
        "\n",
        divider1,
        "\n",
        "Welcome to the quick setup test script!",
        "This script is designed to test that you have correctly setup this project.",
        "Before running this script, ensure you have done the following:",
        "1. Installed the dependencies in requirements.txt",
        "2. Created a .env file and entered your API key and discord token.",
        "3. Entered a template name in the .env file.",
        "4. Entered a home channel in the .env file.",
        "\n",
        divider2,
        "\n",
        "Overview of what we will be doing:",
        "1. Checking that you have installed at least some of the dependencies.",
        "2. Printing out your current settings to ensure that's been setup correctly.",
        "3. Getting a test response from the AI to ensure that you have correctly entered your API key and template name.",
        "4. Trying to send a discord message to ensure that you have correctly setup the discord bot.",
    ]
    print("\n".join(header))
    print("\n")
    input("Press enter to begin....")
    print("\n")
    print(divider1)


def test_dependencies():
    """Test that the dependencies are installed."""
    print("STEP 1: Testing dependencies...")
    print("Attempting to import dependencies...")
    try:
        import discord

        print("Discord dependency imported...")
        import tiktoken

        print("Tiktoken dependency imported...")
        import openai

        print("Openai dependency imported...")
        from dotenv import load_dotenv

    except Exception as e:
       print("An error occurred while importing dependencies.")
       failure(1, "An error occurred while importing dependencies.", "The error message is:", e)

        
    passed(1, "Dependencies imported successfully.", "You can now continue to step 2.")


def make_config():
    from config_maker import make_config

    # code will cause errors unless this file is made, so we will do it here as well as in main.
    print(
        "Making config file(Used to store settings that can be changed during runtime)..."
    )
    make_config()


def obscure_keys(key: str) -> str:
    """Obscure the key."""
    if key is None:
        return "Not Present"
    beginning = key[0:4]
    return f"{beginning}{('*' * (len(key) - 4))} (Length:  { str(len(key))})"


def test_settings():
    """Print out the settings to ensure they are correct."""
    FAIL = f"{RED} STEP 2 FAILED....{RESET}"
    PASS = f"{GREEN} STEP 2 PASSED... {RESET}"
    print("STEP 2: Printing out settings...")

    print(
        "\u001b[1m Please note that your settings in config.ini(or settings changed during run time with commands) take priority to their respective .env variables. This is limited to settings related to autosaving(frequency, enabled, etc), message chunk length, and home channel(unless home channel in config is 0) \u001b[0m"
    )
    time.sleep(1)

    print(
        "Would you like to view your API keys and discord tokens? If you choose no, they will be obscured. (y/n)"
    )

    ans = input(">>> ")
    show_api_keys = False
    if ans.lower().strip() == "y":
        show_api_keys = True
    print("Printing out your settings...")
    print(divider1)
    try:
        from APGCM.settings import SETTINGS_BAG
        from discord_settings import DISCORD_SETTINGS_BAG

        OPENAI_KEY = (
            SETTINGS_BAG.OPENAI_API_KEY
            if show_api_keys
            else obscure_keys(SETTINGS_BAG.OPENAI_API_KEY)
        )
        DISCORD_TOKEN = (
            DISCORD_SETTINGS_BAG.BOT_TOKEN
            if show_api_keys
            else obscure_keys(DISCORD_SETTINGS_BAG.BOT_TOKEN)
        )
        settings_list = [
            f"OPENAI_API_KEY: {OPENAI_KEY}",
            f"DISCORD BOT TOKEN:  {DISCORD_TOKEN}",
            f"TEMPLATE NAME: {SETTINGS_BAG.DEFAULT_TEMPLATE_NAME}",
            f"DEFAULT MODEL: {SETTINGS_BAG.DEFAULT_MODEL}",
            f"HOME CHANNEL: {DISCORD_SETTINGS_BAG.BOT_HOME_CHANNEL}",
            f"IS AUTOSAVING ENABLED: {DISCORD_SETTINGS_BAG.AUTO_SAVING_ENABLED}",
            f"AUTO SAVE FREQUENCY: {DISCORD_SETTINGS_BAG.AUTO_SAVE_FREQUENCY}",
            f"MESSAGE CHUNK LENGTH: {DISCORD_SETTINGS_BAG.MESSAGE_CHUNK_LEN}",
        ]
        print("\n".join(settings_list))
        print("The above should be the settings you entered in the .env file.")

        if input("Do these settings look correct?(Y/N)").lower().strip() != "y":
            print("Please change the settings in the .env file.")
            failure(
                2,
                "Settings are incorrect.",
                "Ensure the settings you have entered in the .env file are correct.",
                f"{BLUE} Tip {RESET}:You may need to select the 'show hidden files' option in your file explorer to see the .env file.",
            )
            
            
    except KeyError as e:
        failure(2, "You have not entered all the required settings in the .env file.", f"{BLUE} Tip {RESET}:You may need to select the 'show hidden files' option in your file explorer to see the .env file.", f"Error message: {e}")

    except Exception as e:
       
        
        print("The error message is:", e)
        failure(
            2,
            "An unknown error has occurred.",
            "This likely means that you have not setup the project correctly, or at least not setup the settings correctly.",
            "Please ensure you have followed the instructions in the README.md file.",
            f"{BLUE} Tip {RESET}:You may need to select the 'show hidden files' option in your file explorer to see the .env file.",
            fail_msg=str(e),
        )
        
    passed(2, "Settings look correct.", "You can now continue to step 3.")

    


def get_test_ai_response():
    """Attempt to get a test response from the AI."""
    FAIL = f"{RED} STEP 3 FAILED....{RESET}"
    PASS = f"{GREEN} STEP 3 PASSED... {RESET}"
    print("STEP 3: Getting a test response from the AI...")
    print(
        "Note that this is an actual request to the API not a mock. You will be charged for this request (But for a message of this length we are sending, it should a fraction of a cent, even with the most expensive model)"
    )
    print(
        "You should see a loading spinner and then a response from the AI thats something like 'All systems are go!'"
    )
    try:
        import APGCM
        import openai

        cw = APGCM.chat_utilities.quick_make_chat_wrapper()
        print("Getting test response...")
        APGCM.chat_utilities.print_test_ai_response(cw)

    except openai.OpenAIError as e:
        print("An error has occurred.")
        failure(3, "OpenAIError", "The error message is:", e, "This means that either your openai api key is incorrect, or you are trying to use a model that you do not have access to.", "If the error message indicates that this is a temporary issue (ie: overloaded API, unknown error), try running this test script again in a few minutes to see if the issue has been resolved.")
        
    except APGCM.exceptions.BadTemplateError as e:
        failure(3, "BadTemplateError", "The error message is:", e, "This means that the template name you have entered in the .env file is incorrect.", "Please ensure that you have entered the template name correctly.", "Either use one of the examples in the .env file's comments or choose one from docs/template_directory.md")
    except APGCM.exceptions.BadTemplateDict as e:
        failure(3, "BadTemplateDict", "It seems that you have attempted to make a custom template but have done it incorrectly.", "Please see the documentation for how to make a custom template.", "The error message is:", e, f"{BLUE} Tip {RESET}: You can run the template reload script in the templates folder in APGCM to reset the template file to the default templates.")
    except APGCM.exceptions.PrettyGoodError as e:
        failure(3, e, "The error message is:", e, "This is almost certainly caused by an issue with the template directory", "Note that the template system does not fully check the templates on loading, there are some values that can only be caught by the various other parts of this project", "Please read the error message and try to fix the issue.", "If you cannot fix the issue, please report it to the developer.")
        
    except Exception as e:
        failure = (3, "An unknown error has occurred.", "The error message is:", e, "If you are not sure what the error message means, please report it to the developer.", "If know what you did wrong(ie modified code, etc), please fix it and try again.", "Otherwise, this is likely an issue with the project itself, and you should report it to the developer.")
    passed(3, "Test response received.", "You can now continue to step 4.")


def test_discord():
    """Explain to user what the test will entail, confirm, and then run the test bot."""
    print("STEP 4: Testing discord...")
    print(
        "This one is a little more complex as there is no easy way to test this without running a bot."
    )

    print(
        "You can either do this now, or just try running the main.py file and see if it works."
    )
    ans = input("Test discord now? (y/n)")
    if ans.lower().strip() == "y":
        print("Starting discord bot...")
        print(
            "\u001b[1m If you see a message in the home channel that says \u001b[35m 'All systems are go!', \u001b[0m \u001b[1m then you have correctly setup the discord bot, and this test has passed. \u001b[0m"
        )

        print(
            "After that, we can be sure that everything is setup correctly and you can start using the program."
        )
        test_bot()
    else:
       skip_test(3, "Skipping discord test.")
       finish()


def test_bot():
    print("\n".join([
        "In order to test the discord bot, we will need to run it.",
        "We are going to send a test message to the home channel",
        "Once you receive the message, reply to it with anything",
        "If the bot replies to your message, and you see a message in the home channel that says 'Got Message: <your message>', then the bot is working correctly.",
        "If you do not your message echoed back to you, then the bot is not working correctly, and likely does not have the correct permissions.",
        "Please see docs/help/setting_up_discord_bot.md for more information on how to setup the discord bot."
    ]))
    try:
        import discord

        from discord_settings import DISCORD_SETTINGS_BAG

        intents = discord.Intents.default()
        intents.members = True
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            home_channel = client.get_channel(DISCORD_SETTINGS_BAG.BOT_HOME_CHANNEL)
            await home_channel.send("All systems are go! Please send a test message to ensure that the bot can read and reply to messages.")
            print("Sent message to home channel.")
        @client.event
        async def on_message(message: discord.Message):
            if message.channel.id != DISCORD_SETTINGS_BAG.BOT_HOME_CHANNEL or message.author.bot:
                return
            print("Got message from home channel.")
            print("Message content:", message.content)
            await message.reply("Got Message: " + message.content)
            passed(4, "If the above message is correct, then the discord bot is working correctly.", "You are now ready to use the program.")
            print("Note, that if you stay on this screen for too long you will get a timeout error. This is normal.")
            print("After exiting the program, you will see another error message, rest assured, this is normal. All tests have passed.")
            finish()
            
            
        client.run(DISCORD_SETTINGS_BAG.BOT_TOKEN)

    except Exception as e:
       failure(4, e, "An error has occurred.", "The error message is:", e, "This is likely an issue with your discord bot token .", "Please ensure that you have entered the discord bot token correctly in the .env file.", "If you are sure its correct, and are still getting this message please report it to the developer.")

def main():
    print_header()
    test_dependencies()
    make_config()
    test_settings()
    get_test_ai_response()
    test_discord()


if __name__ == "__main__":
    main()
