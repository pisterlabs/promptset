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
    makefile = """
    # google test
    GTEST_DIR=/home/adamsl/zero_w_projects/temp/rpi-rgb-led-matrix/tennis-game/googletest
    GTEST_INCDIR=$(GTEST_DIR)/googletest/include
    GTEST_LIBDIR=$(GTEST_DIR)/build/lib
    GTEST_LIBS=-lgtest -lgtest_main

    CFLAGS=-Wall -O3 -g -Wextra -Wno-unused-parameter
    # CXXFLAGS=$(CFLAGS)
    CXXFLAGS=$(CFLAGS) -I$(GTEST_INCDIR)
    # OBJECTS=SetHistoryTextTest.o SetHistoryText.o SetDrawer.o GameLedTranslator.o SubjectManager.o WebLiquidCrystal.o WatchTimer.o Inputs.o TieBreaker.o Mode1Score.o Mode1Functions.o ServeLeds.o Undo.o BatteryTest.o Reset.o SetLeds.o TieLeds.o Mode1WinSequences.o Mode2Functions.o MatchWinSequence.o TennisConstants.o GameLeds.o GameModes.o GameObject.o PinState.o PinInterface.o TranslateConstant.o PointLeds.o Arduino.o CanvasCreator.o FontLoader.o Drawer.o TextDrawer.o GameTimer.o Logger.o History.o GameState.o ScoreBoard.o Player.o tennis-game.o
    BINARIES=tennis-game

    # Where our library resides. It is assumed here that $(RGB_LIB_DISTRIBUTION) has a
    # compiled library in lib/
    RGB_LIB_DISTRIBUTION=..
    RGB_INCDIR=$(RGB_LIB_DISTRIBUTION)/include
    RGB_LIBDIR=$(RGB_LIB_DISTRIBUTION)/lib
    RGB_LIBRARY_NAME=rgbmatrix
    RGB_LIBRARY=$(RGB_LIBDIR)/lib$(RGB_LIBRARY_NAME).a
    # LDFLAGS+=-L$(RGB_LIBDIR) -l$(RGB_LIBRARY_NAME) -lrt -lm -lpthread
    LDFLAGS+=-L$(RGB_LIBDIR) -l$(RGB_LIBRARY_NAME) -L$(GTEST_LIBDIR) $(GTEST_LIBS) -lrt -lm -lpthread

    # To compile image-example
    MAGICK_CXXFLAGS?=$(shell GraphicsMagick++-config --cppflags --cxxflags)
    MAGICK_LDFLAGS?=$(shell GraphicsMagick++-config --ldflags --libs)

    MAIN_OBJECTS=GameWinSequence.o SetWin.o SetHistoryText.o SetDrawer.o GameLedTranslator.o SubjectManager.o WebLiquidCrystal.o WatchTimer.o Inputs.o TieBreaker.o Mode1Score.o Mode1Functions.o ServeLeds.o Undo.o BatteryTest.o Reset.o SetLeds.o TieLeds.o Mode1WinSequences.o Mode2Functions.o MatchWinSequence.o TennisConstants.o GameLeds.o GameModes.o GameObject.o PinState.o PinInterface.o TranslateConstant.o PointLeds.o Arduino.o CanvasCreator.o FontLoader.o Drawer.o TextDrawer.o GameTimer.o Logger.o History.o GameState.o ScoreBoard.o Player.o tennis-game.o
    TEST_OBJECTS=GameState.o Player.o GameState.o GameTimer.o ScoreBoard.o SetDrawer.o Drawer.o CanvasCreator.o FontLoader.o GameTimer.o

    all : $(BINARIES)

    $(RGB_LIBRARY): FORCE
        $(MAKE) -C $(RGB_LIBDIR)

    tennis-game: $(MAIN_OBJECTS)
        $(CXX) $^ -o $@ $(LDFLAGS)

    GameWinSequence.o : GameWinSequence/GameWinSequence.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    SetWin.o : SetWin/SetWin.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    SetDrawer.o : SetDrawer/SetDrawer.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    GameLedTranslator.o : GameLedTranslator/GameLedTranslator.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    CanvasCreator.o : CanvasCreator/CanvasCreator.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    FontLoader.o : FontLoader/FontLoader.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Drawer.o : Drawer/Drawer.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    TextDrawer.o : TextDrawer/TextDrawer.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    GameTimer.o : GameTimer/GameTimer.cpp
        $(CXX) -I$(RGB_INCDIR) -I../Arduino $(CXXFLAGS) -c -o $@ $<

    GameState.o : GameState/GameState.cpp
        $(CXX) -I$(RGB_INCDIR) -I../Arduino $(CXXFLAGS) -c -o $@ $<

    GameObject.o : GameObject/GameObject.cpp
        $(CXX) -I$(RGB_INCDIR) -I../Arduino $(CXXFLAGS) -c -o $@ $<

    GameModes.o : GameModes/GameModes.cpp
        $(CXX) -I$(RGB_INCDIR) -I../Arduino $(CXXFLAGS) -c -o $@ $<

    GameLeds.o : GameLeds/GameLeds.cpp
        $(CXX) -I$(RGB_INCDIR) -I../Arduino $(CXXFLAGS) -c -o $@ $<

    Arduino.o : Arduino/Arduino.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Logger.o : Logger/Logger.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    History.o : History/History.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    ScoreBoard.o : ScoreBoard/ScoreBoard.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Player.o : Player/Player.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    PinInterface.o : PinInterface/PinInterface.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    TranslateConstant.o : TranslateConstant/TranslateConstant.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    PointLeds.o : PointLeds/PointLeds.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    PinState.o : PinState/PinState.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    TennisConstants.o : TennisConstants/TennisConstants.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Mode1Functions.o : Mode1Functions/Mode1Functions.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Mode2Functions.o : Mode2Functions/Mode2Functions.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Mode1WinSequences.o : WinSequences/WinSequences.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    SetLeds.o : SetLeds/SetLeds.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    TieLeds.o : TieLeds/TieLeds.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Reset.o : Reset/Reset.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    MatchWinSequence.o : MatchWinSequence/MatchWinSequence.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    BatteryTest.o : BatteryTest/BatteryTest.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    WebLiquidCrystal.o : WebLiquidCrystal/WebLiquidCrystal.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Undo.o : Undo/Undo.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    ServeLeds.o : ServeLeds/ServeLeds.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Mode1Score.o : Mode1Score/Mode1Score.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    TieBreaker.o : TieBreaker/TieBreaker.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    Inputs.o : Inputs/Inputs.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    WatchTimer.o : WatchTimer/WatchTimer.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    SubjectManager.o : SubjectManager/SubjectManager.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    SetHistoryText.o : SetHistoryText/SetHistoryText.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    tennis-game.o : tennis-game.cpp
        $(CXX) -I$(RGB_INCDIR) $(CXXFLAGS) -c -o $@ $<

    clean:
        rm -f $(MAIN_OBJECTS) $(TEST_OBJECTS) $(BINARIES)

    FORCE:
    .PHONY: FORCE
    """
    # prompt=args.prompt
    prompt = """ make error ``` I need to run the Mode1ScoreTest in this directory.  Please modify the Makefile provided to account for this and let me know how to run the test. ``` """

    directory = args.directory
    model = args.model
    # code_contents = walk_directory(directory)
    code_contents = walk_directory(
        "/home/adamsl/rpi-rgb-led-matrix/tennis-game/GameState")
    # code_contents = walk_directory( "GameState" )

    # Now, `code_contents` is a dictionary that contains the content of all your non-image files
    # You can send this to OpenAI's text-davinci-003 for help

    context = "\n".join(
        f"{path}:\n{contents}" for path, contents in code_contents.items()
    )
    system = "You are an AI expert C++ consultant who is trying to answer questions for the user based on their file system. The user has provided you with the following question and files:"
    prompt = (
        "My files are as follows: "
        + context
        + "\n\n"
        + "Makefile: " + makefile + "\n\n"
        + "My question is as follows: "
        + prompt
    )
    # prompt += (
    #     "\n\nPlease answer my question which files."
    # )
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
