import sys
import os
from time import sleep
# from constants import DEFAULT_DIR, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, EXTENSION_TO_SKIP
import argparse
def read_file(filename):
    with open(filename, "r") as file:
        return file.read()


def walk_directory(directory):
    image_extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".svg",
        ".ico",
        ".tif",
        ".tiff",
        ".txt",
    ]
    # get current working directory

    code_contents = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not any(file.endswith(ext) for ext in image_extensions):
                try:
                    relative_filepath = os.path.relpath(
                        os.path.join(root, file), directory
                    )
                    code_contents[relative_filepath] = read_file(
                        os.path.join(root, file)
                    )
                except Exception as e:
                    code_contents[
                        relative_filepath
                    ] = f"Error reading file {file}: {str(e)}"
    return code_contents


def main(args):
    # prompt=args.prompt
    prompt= """ make error ``` adamsl@DESKTOP-BKHEBT0:~/rpi-rgb-led-matrix/tennis-game$ make tennis-game
g++ -I../include -I../Arduino -Wall -O3 -g -Wextra -Wno-unused-parameter -I/home/adamsl/zero_w_projects/temp/rpi-rgb-led-matrix/tennis-game/googletest/googletest/include -c -o GameState.o GameState/GameState.cpp
In file included from GameState/GameState.cpp:1:
GameState/GameState.h:103:10: error: ‘void GameState::setPlayer1SetHistory(std::map<int, int>)’ cannot be overloaded with ‘void GameState::setPlayer1SetHistory(std::map<int, int>)’
  103 |     void setPlayer1SetHistory(std::map<int, int> player1_set_history) override;
      |          ^~~~~~~~~~~~~~~~~~~~
GameState/GameState.h:17:10: note: previous declaration ‘void GameState::setPlayer1SetHistory(std::map<int, int>)’
   17 |     void setPlayer1SetHistory(std::map<int, int> set_history) override;
      |          ^~~~~~~~~~~~~~~~~~~~
GameState/GameState.h:104:24: error: ‘std::map<int, int> GameState::getPlayer1SetHistory()’ cannot be overloaded with ‘std::map<int, int> GameState::getPlayer1SetHistory()’
  104 |     std::map<int, int> getPlayer1SetHistory() override;
      |                        ^~~~~~~~~~~~~~~~~~~~
GameState/GameState.h:18:24: note: previous declaration ‘std::map<int, int> GameState::getPlayer1SetHistory()’
   18 |     std::map<int, int> getPlayer1SetHistory() override;
      |                        ^~~~~~~~~~~~~~~~~~~~
GameState/GameState.h:105:10: error: ‘void GameState::setPlayer2SetHistory(std::map<int, int>)’ cannot be overloaded with ‘void GameState::setPlayer2SetHistory(std::map<int, int>)’
  105 |     void setPlayer2SetHistory(std::map<int, int> player2_set_history) override;
      |          ^~~~~~~~~~~~~~~~~~~~
GameState/GameState.h:19:10: note: previous declaration ‘void GameState::setPlayer2SetHistory(std::map<int, int>)’
   19 |     void setPlayer2SetHistory(std::map<int, int> set_history) override;
      |          ^~~~~~~~~~~~~~~~~~~~
GameState/GameState.h:106:24: error: ‘std::map<int, int> GameState::getPlayer2SetHistory()’ cannot be overloaded with ‘std::map<int, int> GameState::getPlayer2SetHistory()’
  106 |     std::map<int, int> getPlayer2SetHistory() override;
      |                        ^~~~~~~~~~~~~~~~~~~~
GameState/GameState.h:20:24: note: previous declaration ‘std::map<int, int> GameState::getPlayer2SetHistory()’
   20 |     std::map<int, int> getPlayer2SetHistory() override;
      |                        ^~~~~~~~~~~~~~~~~~~~
make: *** [Makefile:66: GameState.o] Error 1 ``` """

    directory= args.directory
    model=args.model
    # code_contents = walk_directory(directory)
    code_contents = walk_directory( "/home/adamsl/rpi-rgb-led-matrix/tennis-game/GameState" )
    # code_contents = walk_directory( "GameState" )

    # Now, `code_contents` is a dictionary that contains the content of all your non-image files
    # You can send this to OpenAI's text-davinci-003 for help

    context = "\n".join(
        f"{path}:\n{contents}" for path, contents in code_contents.items()
    )
    system = "You are an AI debugger who is trying to debug a program for a user based on their file system. The user has provided you with the following files and their contents, finally folllowed by the error message or issue they are facing."
    prompt = (
        "My files are as follows: "
        + context
        + "\n\n"
        + "My issue is as follows: "
        + prompt
    )
    prompt += (
        "\n\nGive me ideas for what could be wrong and what fixes to do in which files."
    )
    res = generate_response(system, prompt, model)
    # print res in teal
    print("\033[96m" + res + "\033[0m")


def generate_response(system_prompt, user_prompt, model="gpt-3.5-turbo-16k", *args):
    import openai

    # Set up your OpenAI API credentials
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = "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"

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    # loop thru each arg and add it to messages alternating role between "assistant" and "user"
    role = "assistant"
    for value in args:
        messages.append({"role": role, "content": value})
        role = "user" if role == "assistant" else "assistant"

    params = {
        # "model": model,
        "model": "gpt-3.5-turbo-16k",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0,
    }

    # Send the API request
    keep_trying = True
    while keep_trying:
        try:
            response = openai.ChatCompletion.create(**params)
            keep_trying = False
        except Exception as e:
            # e.g. when the API is too busy, we don't want to fail everything
            print("Failed to generate response. Error: ", e)
            sleep(30)
            print("Retrying...")

    # Get the reply from the API response
    reply = response.choices[0]["message"]["content"]
    return reply


if __name__ == "__main__":
    DEFAULT_DIR = "player_debug"
    DEFAULT_MODEL = "gpt-3.5-turbo-16k"
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "prompt",
    #     help="The prompt to use for the AI. This should be the error message or issue you are facing.",
        
    # )
    parser.add_argument(
        "--directory",
        "-d",
        help="The directory to use for the AI. This should be the directory containing the files you want to debug.",
        default=DEFAULT_DIR,
    )
    parser.add_argument(
        "--model",
        "-m",
        help="The model to use for the AI. This should be the model ID of the model you want to use.",
        default=DEFAULT_MODEL,
    )
    args = parser.parse_args()
    main(args)