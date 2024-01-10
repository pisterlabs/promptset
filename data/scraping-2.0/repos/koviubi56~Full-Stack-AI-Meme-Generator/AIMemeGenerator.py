#!/usr/bin/env python3
"""
AI Meme Generator.

Creates start-to-finish memes using various AI service APIs. OpenAI's chatGPT
to generate the meme text and image prompt, and several optional image
generators for the meme picture. Then combines the meme text and image into a
meme using Pillow.

Originally created by ThioJoe <github.com/ThioJoe/Full-Stack-AI-Meme-Generator>
Modified by Koviubi56 in 2023.

Copyright (C) 2023  Koviubi56

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
__version__ = "2.0.0-beta.1"

import abc
import argparse
import base64
import dataclasses
import datetime
import io
import os
import pathlib
import platform
import re
import shutil
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import colorama
import termcolor

try:
    import tomllib  # novermin
except Exception:
    import tomli as tomllib

colorama.init()

SETTINGS_FILE_NAME = "settings.toml"
DEFAULT_SETTINGS_FILE_NAME = "settings_default.toml"
API_KEYS_FILE_NAME = "api_keys.toml"
DEFAULT_API_KEYS_FILE_NAME = "api_keys_empty.toml"

FORMAT_INSTRUCTIONS = (
    "You are a meme generator with the following formatting instructions."
    " Each meme will consist of text that will appear at the top, and an"
    " image to go along with it. The user will send you a message with a"
    " general theme or concept on which you will base the meme. The user"
    ' may choose to send you a text saying something like "anything" or'
    ' "whatever you want", or even no text at all, which you should not'
    " take literally, but take to mean they wish for you to come up with"
    " something yourself.  The memes don't necessarily need to start with"
    ' "when", but they can. In any case, you will respond with two things:'
    " First, the text of the meme that will be displayed in the final"
    " meme. Second, some text that will be used as an image prompt for an"
    " AI image generator to generate an image to also be used as part of"
    " the meme. You must respond only in the format as described next,"
    " because your response will be parsed, so it is important it conforms"
    ' to the format. The first line of your response should be: "Meme'
    ' Text: " followed by the meme text. The second line of your response'
    ' should be: "Image Prompt: " followed by the image prompt text.  ---'
    " Now here are additional instructions... "
    "Next are instructions for the overall approach you should take to"
    " creating the memes. Interpret as best as possible:"
    " {} | "
    "Next are any special instructions for the image prompt. For example,"
    ' if the instructions are "the images should be photographic style",'
    ' your prompt may append ", photograph" at the end, or begin with'
    ' "photograph of". It does not have to literally match the instruction'
    " but interpret as best as possible: {} | Now come up with a meme"
    " according to the previous instructions! "
)


class MemeGeneratorError(RuntimeError):
    """Base class for all AIMemeGenerator exceptions."""


@dataclasses.dataclass
class NoFontFileError(MemeGeneratorError):
    """Could not find the font file."""

    font_file: str

    def __str__(self) -> str:
        return (
            f"Font file {self.font_file!r} not found. Please add the font file"
            " to the same folder as this script. Or set the variable above to"
            " the name of a font file in the system font folder."
        )


@dataclasses.dataclass
class MissingAPIKeyError(MemeGeneratorError):
    """A required API key is missing."""

    api: Literal["openai", "stability", "clipdrop", "ALL"]

    def __str__(self) -> str:
        if self.api == "ALL":
            return (
                "No API keys found. Please provide one of function arguments,"
                " command line arguments, or api keys file."
            )
        return (
            f"{self.api} is expecting an API key, but no {self.api} API key"
            " was found in the api_keys.toml file."
        )


@dataclasses.dataclass
class InvalidTextPlatformError(MemeGeneratorError):
    """Invalid text platform."""

    text_platform: str

    def __str__(self) -> str:
        return f"Invalid text platform {self.text_platform!r}."


@dataclasses.dataclass
class InvalidImagePlatformError(MemeGeneratorError):
    """Invalid image platform."""

    image_platform: str

    def __str__(self) -> str:
        return f"Invalid image platform {self.image_platform!r}."


@dataclasses.dataclass
class APIKeys:
    """
    The API keys.

    Args:
        openai_key (str): OpenAI API key.
        clipdrop_key (Optional[str]): ClipDrop API key.
        stability_key (Optional[str]): Stability API key.
    """

    openai_key: str
    clipdrop_key: Optional[str]
    stability_key: Optional[str]


@dataclasses.dataclass
class Meme:
    """
    A dictionary containing the meme's text and image prompt.

    Args:
        meme_text (str): The meme's text.
        image_prompt (str): The image prompt.
    """

    meme_text: str
    image_prompt: str


@dataclasses.dataclass
class FullMeme:
    """
    A full meme.

    Args:
        meme_text (str): The meme's text.
        image_prompt (str): The image's prompt.
        virtual_meme_file (io.BytesIO): The virtual meme image file.
        file (pathlib.Path): The meme image file.
    """

    meme_text: str
    image_prompt: str
    virtual_meme_file: io.BytesIO
    file: pathlib.Path


# Parse the arguments at the start of the script
parser = argparse.ArgumentParser()
parser.add_argument(
    "--text-generation-service",
    help='The text generation service/platform to use. Must be one of "openai"'
    ' or "gpt4all"',
    choices={"openai", "gpt4all"},
)
parser.add_argument("--text-model", help="The text model to use")
parser.add_argument("--openai-key", help="OpenAI API key")
parser.add_argument("--clipdrop-key", help="ClipDrop API key")
parser.add_argument("--stability-key", help="Stability AI API key")
parser.add_argument(
    "--user-prompt",
    help="A meme subject or concept to send to the chat bot. If not specified,"
    " the user will be prompted to enter a subject or concept.",
)
parser.add_argument(
    "--meme-count",
    help="The number of memes to create. If using arguments and not specified,"
    " the default is 1.",
)
parser.add_argument(
    "--image-platform",
    help="The image platform to use. If using arguments and not specified, the"
    " default is 'clipdrop'. Possible options: 'openai', 'stability',"
    " 'clipdrop'",
    choices={"openai", "stability", "clipdrop"},
)
parser.add_argument(
    "--temperature",
    help="The temperature to use for the chat bot. If using arguments and not"
    " specified, the default is 1.0",
)
parser.add_argument(
    "--basic-instructions",
    help="The basic instructions to use for the chat bot. If using arguments"
    " and not specified, default will be used.",
)
parser.add_argument(
    "--image-special-instructions",
    help="The image special instructions to use for the chat bot. If using"
    " arguments and not specified, default will be used",
)
# These don't need to be specified as true/false, just specifying them will
# set them to true
parser.add_argument(
    "--no-user-input",
    action="store_true",
    help="Will prevent any user input prompts, and will instead use default"
    " values or other arguments.",
)
parser.add_argument(
    "--no-file-save",
    action="store_true",
    help="If specified, the meme will not be saved to a file, and only"
    " returned as virtual file part of memeResultsDictsList.",
)


def search_for_file(
    directory: pathlib.Path, file_name: str
) -> Optional[pathlib.Path]:
    """
    Search for the file `file_name` within `directory`.

    Args:
        directory (pathlib.Path): The directory to search in.
        file_name (str): The file's name to look for.

    Returns:
        Optional[pathlib.Path]: The first file that matched `file_name` or
        None.
    """
    try:
        return next(directory.rglob(file_name))
    except StopIteration:
        return None


def search_for_file_in_directories(
    directories: Iterable[pathlib.Path], file_name: str
) -> Optional[pathlib.Path]:
    """
    Search for the file `file_name` within `directories`.

    Args:
        directories (Iterable[pathlib.Path]): The directories to search in.
        file_name (str): The file's name to look for.

    Returns:
        Optional[pathlib.Path]: The first file that matched `file_name` or
        None.
    """
    for directory in directories:
        file = search_for_file(directory, file_name)
        if file:
            return file
    return None


def check_font(font_file_name: str) -> pathlib.Path:
    """
    Check for font file in current directory, then check for font file in Fonts
    folder, warn user and exit if not found.

    Args:
        font_file_name (str): The font file's name.
        no_user_input (bool): Don't ask for user input.

    Returns:
        pathlib.Path: The font file.
    """
    # Check for font file in current directory
    termcolor.cprint("Checking the font...", "black")
    path = pathlib.Path(font_file_name)
    if path.exists():
        return path

    if platform.system() == "Windows":
        # Check for font file in Fonts folder (Windows)
        file = pathlib.Path(os.environ["WINDIR"], "Fonts", font_file_name)
    elif platform.system() == "Linux":
        # Check for font file in font directories (Linux)
        font_directories = [
            pathlib.Path("/usr/share/fonts"),
            pathlib.Path("~/.fonts").expanduser(),
            pathlib.Path("~/.local/share/fonts").expanduser(),
            pathlib.Path("/usr/local/share/fonts"),
        ]
        file = search_for_file_in_directories(font_directories, font_file_name)
    elif (
        platform.system() == "Darwin"
    ):  # Darwin is the underlying system for macOS
        # Check for font file in font directories (macOS)
        font_directories = [
            pathlib.Path("/Library/Fonts"),
            pathlib.Path("~/Library/Fonts").expanduser(),
        ]
        file = search_for_file_in_directories(font_directories, font_file_name)
    else:
        file = None

    # Warn user and exit if not found
    if (not file) or (not file.exists()):
        raise NoFontFileError(font_file_name)
    # Return the font file path
    return file


def get_config(
    config_file: pathlib.Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of the config file.

    Args:
        config_file (pathlib.Path): The config file.

    Returns:
        Dict[str, Dict[str, Any]]: The settings read from the file.
    """
    with config_file.open("rb") as file:
        return tomllib.load(file)


def get_assets_file(file_name: str) -> pathlib.Path:
    """
    Get `assets/file_name`

    Args:
        file_name (str): The file's name.

    Returns:
        pathlib.Path: The asset file.
    """
    if hasattr(sys, "_MEIPASS"):  # If running as a pyinstaller bundle
        return pathlib.Path(sys._MEIPASS, file_name).resolve()  # noqa: SLF001
    return pathlib.Path(__file__, "..", "assets", file_name).resolve()


def get_settings(no_user_input: bool) -> Dict[str, Dict[str, Any]]:
    """
    Get the settings. Create the file if it doesn't exist.

    Args:
        no_user_input (bool): Don't ask for user input

    Returns:
        Dict[str, Dict[str, Any]]: The settings.
    """
    termcolor.cprint("Getting settings...", "black")
    file = pathlib.Path(SETTINGS_FILE_NAME).resolve()

    if not file.exists():
        termcolor.cprint(
            f"WARNING: The config file {file} does not exist!", "yellow"
        )
        if (not no_user_input) and (
            input(
                "Create the config file with default settings? [Y/n] "
            ).lower()
            != "n"
        ):
            file_to_copy_path = get_assets_file(DEFAULT_SETTINGS_FILE_NAME)
            shutil.copyfile(file_to_copy_path, SETTINGS_FILE_NAME)
            termcolor.cprint(
                f"Default {SETTINGS_FILE_NAME} file created. You can use it"
                " going forward to change more advanced settings if you want.",
                "cyan",
            )
        else:
            termcolor.cprint("WARNING: Using default settings", "yellow")
            return {}

    # Try to get settings file, if fails, use default settings
    try:
        settings = get_config(file)
    except Exception:
        termcolor.cprint(traceback.format_exc(), "yellow")
        termcolor.cprint(
            "WARNING: Could not read settings file. Using default settings"
            " instead.",
            "yellow",
        )
        settings = {}

    termcolor.cprint(f"Config will be {settings}", "black")
    return settings


@dataclasses.dataclass
class TextABC(abc.ABC):
    user_entered_prompt: str
    basic_instructions: str
    image_special_instructions: str

    @abc.abstractmethod
    def generate_response(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self) -> None:
        return  # kinda optional; not gonna raise

    @classmethod
    def create_initialize_and_generate(cls, **kwargs: Any) -> str:
        instance = cls(**kwargs)
        instance.initialize()
        return instance.generate_response()


@dataclasses.dataclass
class OpenAIText(TextABC):
    api_key: str
    text_model: str
    temperature: float

    def _set_api_key(self) -> None:
        import openai

        openai.api_key = self.api_key

    def initialize(self) -> None:
        if not self.api_key:
            raise MissingAPIKeyError("openai")
        self._set_api_key()
        system_prompt = construct_system_prompt(
            self.basic_instructions, self.image_special_instructions
        )
        self.conversation = [{"role": "system", "content": system_prompt}]

    def _chat_completion_create(self) -> Any:
        import openai

        return openai.ChatCompletion.create(
            model=self.text_model,
            messages=self.conversation,
            temperature=self.temperature,
        )

    def generate_response(self) -> str:
        from openai.error import InvalidRequestError, RateLimitError

        # Prepare to send request along with context by appending user message
        # to previous conversation
        self.conversation.append(
            {"role": "user", "content": self.user_entered_prompt}
        )

        termcolor.cprint(
            f"  Sending request to write meme to model {self.text_model}...",
            "cyan",
        )
        try:
            chat_response = self._chat_completion_create()
        except RateLimitError:
            termcolor.cprint(
                "\nERROR! See below hint and traceback!", "yellow"
            )
            termcolor.cprint(
                "hint: Did you setup payment? See <https://openai.com/pricing>",
                "cyan",
            )
            raise
        except InvalidRequestError as error:
            termcolor.cprint(
                "\nERROR! See below hint and traceback!", "yellow"
            )
            if "The model" in str(error) and "does not exist" in str(error):
                if str(error) == "The model `gpt-4` does not exist":
                    termcolor.cprint(
                        "hint: You do not have access to the GPT-4 model yet.",
                        "cyan",
                    )
                    termcolor.cprint(
                        "hint: You can see more about the current GPT-4"
                        " requirements here: <https://help.openai.com/en/articles"
                        "/7102672-how-can-i-access-gpt-4>",
                        "cyan",
                    )
                    termcolor.cprint(
                        "hint: Also ensure your country is supported:"
                        " <https://platform.openai.com/docs/supported-countries>",
                        "cyan",
                    )
                    termcolor.cprint(
                        "hint: You can try the 'gpt-3.5-turbo' model instead."
                        " See more here: <https://platform.openai.com/docs"
                        "/models/overview>",
                        "cyan",
                    )
                else:
                    termcolor.cprint(
                        "hint: Either the model name is incorrect, or you do"
                        " not have access to it.",
                        "cyan",
                    )
                    termcolor.cprint(
                        "hint: See this page to see the model names to use in"
                        " the API: <https://platform.openai.com/docs/models/overview>",
                        "cyan",
                    )
            raise

        return chat_response.choices[0].message.content


@dataclasses.dataclass
class GPT4AllText(TextABC):
    text_model: str  # "ggml-model-gpt4all-falcon-q4_0.bin"
    temperature: float

    def _get_model(self) -> Any:
        from gpt4all import GPT4All

        return GPT4All(model_name=self.text_model)

    def initialize(self) -> None:
        if self.text_model.startswith("gpt-"):
            termcolor.cprint(
                "WARNING: It looks like you forgot to change the text_model in"
                " the config file! Please edit the config file and try again.",
                "yellow",
            )
        elif self.text_model != "ggml-model-gpt4all-falcon-q4_0.bin":
            termcolor.cprint(
                'WARNING: Only the "ggml-model-gpt4all-falcon-q4_0.bin"'
                " GPT4All model has been tested! Others might not work!"
                " Proceed with caution!",
                "yellow",
            )
        termcolor.cprint(
            "WARNING! By using GPT4All an ~8 GB AI model will be downloaded"
            " (will be cached at ~/.cache)!"
            "\nProceed with caution, press CTRL+C to abort!",
            "yellow",
        )
        _model_start = time.perf_counter()
        self.model = self._get_model()
        _model_end = time.perf_counter()
        termcolor.cprint(
            f"Initialized model in {_model_end - _model_start} seconds",
            "black",
        )

    def generate_response(self) -> str:
        termcolor.cprint(
            "Generating meme text, please wait ~2 minutes...", "black"
        )
        _generate_start = time.perf_counter()
        output = self.model.generate(
            prompt=construct_system_prompt(
                self.basic_instructions, self.image_special_instructions
            ),
            temp=self.temperature,
        )
        _generate_end = time.perf_counter()
        termcolor.cprint(
            f"Generated response in {_generate_end - _generate_start} seconds",
            "black",
        )
        return output


@dataclasses.dataclass
class ImageABC(abc.ABC):
    image_prompt: str

    @abc.abstractmethod
    def generate_image(self) -> io.BytesIO:
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self) -> None:
        return  # kinda optional; not gonna raise

    @classmethod
    def create_initialize_and_generate(cls, **kwargs: Any) -> io.BytesIO:
        instance = cls(**kwargs)
        instance.initialize()
        return instance.generate_image()


@dataclasses.dataclass
class OpenAIImage(ImageABC):
    api_key: str

    def _set_api_key(self) -> None:
        import openai

        openai.api_key = self.api_key

    def initialize(self) -> None:
        if not self.api_key:
            raise MissingAPIKeyError("openai")
        self._set_api_key()

    def _image_create(self) -> Any:
        import openai

        return openai.Image.create(
            prompt=self.image_prompt,
            n=1,
            size="512x512",
            response_format="b64_json",
        )

    def generate_image(self) -> io.BytesIO:
        openai_response = self._image_create()
        # Convert image data to virtual file
        return io.BytesIO(
            base64.b64decode(openai_response["data"][0]["b64_json"])
        )


@dataclasses.dataclass
class StabilityImage(ImageABC):
    api_key: str

    def _get_interface(self) -> Any:
        from stability_sdk import client

        return client.StabilityInference(
            key=self.api_key,
            verbose=True,
            engine="stable-diffusion-xl-1024-v0-9",
        )

    def initialize(self) -> None:
        if not self.api_key:
            raise MissingAPIKeyError("stability")

        self.stability_api = self._get_interface()

    def generate_image(self) -> io.BytesIO:
        import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation  # noqa: E501

        # Set up our initial generation parameters.
        stability_response = self.stability_api.generate(
            prompt=self.image_prompt,
            steps=30,
            cfg_scale=7.0,
            width=1024,
            height=1024,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2M,
        )

        # Set up our warning to print to the console if the adult content
        # classifier is tripped. If adult content classifier is not tripped,
        # save generated images.
        for resp in stability_response:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    raise ValueError(
                        "Your request activated the API's safety filters and"
                        " could not be processed. Please modify the prompt and"
                        " try again."
                    )
                return io.BytesIO(artifact.binary)
        raise NotImplementedError("\\U0001f480")  # :skull:


@dataclasses.dataclass
class ClipdropImage(ImageABC):
    api_key: str

    def initialize(self) -> None:
        if not self.api_key:
            raise MissingAPIKeyError("clipdrop")

    def _request(self) -> Any:
        import requests

        return requests.post(
            "https://clipdrop-api.co/text-to-image/v1",
            files={"prompt": (None, self.image_prompt, "text/plain")},
            headers={"x-api-key": self.api_key},
            timeout=60,
        )

    def generate_image(self) -> io.BytesIO:
        r = self._request()
        r.raise_for_status()
        return io.BytesIO(r.content)


def get_api_keys(
    args: Optional[argparse.Namespace] = None, no_user_input: bool = False
) -> APIKeys:
    """
    Get API key constants from config file or command line arguments.

    Args:
        args (Optional[argparse.Namespace], optional): The command line
        namespace. Defaults to None.
        no_user_input (bool, optional): Don't ask for user input. Defaults to
        False.

    Returns:
        APIKeys: The API keys.
    """
    # Checks if api_keys.toml file exists, if not create empty one from default
    file = pathlib.Path(API_KEYS_FILE_NAME)

    # Default values
    openai_key, clipdrop_key, stability_key = None, None, None

    # Checks if any arguments are not None, and uses those values if so
    if args and any((args.openai_key, args.clipdrop_key, args.stability_key)):
        termcolor.cprint(
            "Getting all API keys from command line arguments...", "cyan"
        )
        openai_key = args.openai_key if args.openai_key else openai_key
        clipdrop_key = args.clipdrop_key if args.clipdrop_key else clipdrop_key
        stability_key = (
            args.stability_key if args.stability_key else stability_key
        )
        return APIKeys(openai_key, clipdrop_key, stability_key)

    termcolor.cprint("Getting API keys from file...", "cyan")
    if file.exists():
        keys_dict = get_config(file).get("keys", {})
        openai_key = keys_dict.get("openai", None) or None
        clipdrop_key = keys_dict.get("clipdrop", None) or None
        stability_key = keys_dict.get("stabilityai", None) or None
        return APIKeys(openai_key, clipdrop_key, stability_key)

    termcolor.cprint(
        f"WARNING: The api keys file {file} does not exist!", "yellow"
    )
    if (not no_user_input) and (
        input("Create the api keys file with default settings? [Y/n] ").lower()
        != "n"
    ):
        file_to_copy_path = get_assets_file(DEFAULT_API_KEYS_FILE_NAME)
        shutil.copyfile(file_to_copy_path, API_KEYS_FILE_NAME)
        termcolor.cprint(
            "Please add your API keys to the API Keys file.",
            "cyan",
        )
        input("Press Enter to exit...")
        sys.exit(1)
    termcolor.cprint(
        "ERROR: Could not get the API keys from neither the function"
        " arguments, the command line arguments, nor the api keys file. Please"
        " provide one of them!",
        "red",
    )
    raise MissingAPIKeyError("ALL")


def set_file_path(
    base_name: str, output_directory: pathlib.Path
) -> pathlib.Path:
    """
    Sets the name and path of the file to be used.

    Args:
        base_name (str): The base name for the file.
        output_directory (pathlib.Path): The directory to put the file in.

    Returns:
        pathlib.Path: The new file.
    """
    # Generate a timestamp string to append to the file name
    timestamp = datetime.datetime.now().strftime("%f")  # noqa: DTZ005

    # If the output folder does not exist, create it
    output_directory.mkdir(parents=True, exist_ok=True)

    return pathlib.Path(output_directory, f"{base_name}_{timestamp}.png")


def construct_system_prompt(
    basic_instructions: str, image_special_instructions: str
) -> str:
    """
    Construct the system prompt for the chat bot.

    Args:
        basic_instructions (str): The basic AI instructions.
        image_special_instructions (str): The special AI instructions.

    Returns:
        str: The system prompt.
    """
    return FORMAT_INSTRUCTIONS.format(
        basic_instructions, image_special_instructions
    )


def parse_meme(message: str) -> Optional[Meme]:
    """
    Gets the meme text and image prompt from the message sent by the chat
    bot.

    Args:
        message (str): The AI message.

    Returns:
        Optional[MemeDict]: The meme dictionary or None.
    """
    # The regex pattern to match
    pattern = r"\s*Meme Text: (\"(.*?)\"|(.*?))\s*Image Prompt: (.*?)$"

    match = re.search(pattern, message, re.DOTALL)

    if match:
        # If meme text is enclosed in quotes it will be in group 2, otherwise,
        # it will be in group 3.
        meme_text = (
            match.group(2) if match.group(2) is not None else match.group(3)
        )

        return Meme(meme_text=meme_text, image_prompt=match.group(4))
    return None


def create_meme(
    image_path: io.BytesIO,
    top_text: str,
    file_path: pathlib.Path,
    font_file: pathlib.Path,
    no_file_save: bool = False,
    min_scale: float = 0.05,
    buffer_scale: float = 0.03,
    font_scale: float = 1,
) -> io.BytesIO:
    """
    Create the meme image.

    Args:
        image_path (io.BytesIO): The virtual image file.
        top_text (str): Top text.
        file_path (pathlib.Path): The file to write the image to.
        font_file (pathlib.Path): The font file.
        no_file_save (bool, optional): Don't save the file to `file_path`.
        Defaults to False.
        min_scale (float, optional): Minimum scale. Defaults to 0.05.
        buffer_scale (float, optional): Buffer scale. Defaults to 0.03.
        font_scale (float, optional): Font scale. Defaults to 1.

    Returns:
        io.BytesIO: The virtual image file.
    """
    termcolor.cprint("  Creating meme image...", "cyan")
    from PIL import Image, ImageDraw, ImageFont

    # Load the image. Can be a path or a file-like object such as IO.BytesIO
    # virtual file
    image = Image.open(image_path)

    # Calculate buffer size based on buffer_scale
    buffer_size = int(buffer_scale * image.width)

    # Get a drawing context
    d = ImageDraw.Draw(image)

    # Split the text into words
    words = top_text.split()

    # Initialize the font size and wrapped text
    font_size = int(font_scale * image.width)
    fnt = ImageFont.truetype(str(font_file), font_size)
    wrapped_text = top_text

    # Try to fit the text on a single line by reducing the font size
    while (
        d.textbbox((0, 0), wrapped_text, font=fnt)[2]
        > image.width - 2 * buffer_size
    ):
        font_size *= 0.9  # Reduce the font size by 10%
        if font_size < min_scale * image.width:
            # If the font size is less than the minimum scale, wrap the text
            lines = [words[0]]
            for word in words[1:]:
                new_line = (lines[-1] + " " + word).rstrip()
                if (
                    d.textbbox((0, 0), new_line, font=fnt)[2]
                    > image.width - 2 * buffer_size
                ):
                    lines.append(word)
                else:
                    lines[-1] = new_line
            wrapped_text = "\n".join(lines)
            break
        fnt = ImageFont.truetype(str(font_file), int(font_size))

    # Calculate the bounding box of the text
    textbbox_val = d.multiline_textbbox((0, 0), wrapped_text, font=fnt)

    # Create a white band for the top text, with a buffer equal to 10% of the
    # font size
    band_height = (
        textbbox_val[3]
        - textbbox_val[1]
        + int(font_size * 0.1)
        + 2 * buffer_size
    )
    band = Image.new("RGBA", (image.width, band_height), (255, 255, 255, 255))

    # Draw the text on the white band
    d = ImageDraw.Draw(band)

    # The midpoint of the width and height of the bounding box
    text_x = band.width // 2
    text_y = band.height // 2

    d.multiline_text(
        (text_x, text_y),
        wrapped_text,
        font=fnt,
        fill=(0, 0, 0, 255),
        anchor="mm",
        align="center",
    )

    # Create a new image and paste the band and original image onto it
    new_img = Image.new("RGBA", (image.width, image.height + band_height))
    new_img.paste(band, (0, 0))
    new_img.paste(image, (0, band_height))

    if not no_file_save:
        # Save the result to a file
        new_img.save(file_path)

    # Return image as virtual file
    virtual_meme_file = io.BytesIO()
    new_img.save(virtual_meme_file, format="PNG")

    return virtual_meme_file


def text_generation_request(
    api_keys: APIKeys,
    user_entered_prompt: str,
    basic_instructions: str,
    image_special_instructions: str,
    platform: str,
    text_model: str,
    temperature: float,
) -> str:
    """
    Create the text.

    Args:
        api_keys (APIKeys): The API keys.
        user_entered_prompt (str): The user entered prompt.
        basic_instructions (str): Basic instructions.
        image_special_instructions (str): Image special instructions.
        platform (str): The image platform to use.
        text_model (str): The text model to use.
        temperature (float): The temperature.

    Returns:
        str: The response.
    """
    if platform == "openai":
        return OpenAIText.create_initialize_and_generate(
            user_entered_prompt=user_entered_prompt,
            basic_instructions=basic_instructions,
            image_special_instructions=image_special_instructions,
            api_key=api_keys.openai_key,
            text_model=text_model,
            temperature=temperature,
        )

    if platform == "gpt4all":
        return GPT4AllText.create_initialize_and_generate(
            user_entered_prompt=user_entered_prompt,
            basic_instructions=basic_instructions,
            image_special_instructions=image_special_instructions,
            text_model=text_model,
            temperature=temperature,
        )

    raise InvalidTextPlatformError(platform)


def image_generation_request(
    api_keys: APIKeys,
    image_prompt: str,
    platform: str,
) -> io.BytesIO:
    """
    Create the image.

    Args:
        api_keys (APIKeys): The API keys.
        image_prompt (str): The image platform to use.
        platform (str): The platform to use.

    Returns:
        io.BytesIO: The virtual image file.
    """
    if platform == "openai":
        return OpenAIImage.create_initialize_and_generate(
            image_prompt=image_prompt, api_key=api_keys.openai_key
        )

    if platform == "stability":
        return StabilityImage.create_initialize_and_generate(
            image_prompt=image_prompt, api_key=api_keys.stability_key
        )

    if platform == "clipdrop":
        return ClipdropImage.create_initialize_and_generate(
            image_prompt=image_prompt, api_key=api_keys.clipdrop_key
        )

    raise InvalidImagePlatformError(platform)


def generate(
    text_generation_service: str = "openai",
    text_model: str = "gpt-4",
    temperature: float = 1.0,
    basic_instructions: str = "You should come up with some random funny memes"
    " that are clever and original, and not cliche or lame.",
    image_special_instructions: str = "The images should be photographic and"
    " related to the meme.",
    user_entered_prompt: str = "anything",
    meme_count: int = 1,
    image_platform: str = "openai",
    font_file_name: str = "arial.ttf",
    base_file_name: str = "meme",
    output_directory: Union[str, pathlib.Path] = "Outputs",
    openai_key: Optional[str] = None,
    stability_key: Optional[str] = None,
    clipdrop_key: Optional[str] = None,
    no_user_input: bool = False,
    no_file_save: bool = False,
) -> List[FullMeme]:
    """
    Generate the memes.

    Args:
        text_generation_service (str, optional): The text generation
        service/platform to use. Defaults to "openai".
        text_model (str, optional): The text model to use. Defaults to "gpt-4".
        temperature (float, optional): The temperature (randomness). Defaults
        to 1.0.
        basic_instructions (str, optional): The basic instructions. Has
        default.
        image_special_instructions (str, optional): The image special
        instructions. Has default.
        user_entered_prompt (str, optional): The user entered prompt. Defaults
        to "anything".
        meme_count (int, optional): The amount of memes to generate. Defaults
        to 1.
        image_platform (str, optional): The image platform to use. Must be one
        of "openai", "stability", and "clipdrop". Defaults to "openai".
        font_file_name (str, optional): The font file's name to use. Defaults
        to "arial.ttf".
        base_file_name (str, optional): The base file name for the images.
        Defaults to "meme".
        output_directory (pathlib.Path, optional): The directory to put the
        images into. Defaults to pathlib.Path("Outputs").
        openai_key (Optional[str], optional): The OpenAI API key. Defaults to
        None.
        stability_key (Optional[str], optional): The stability API key.
        Defaults to None.
        clipdrop_key (Optional[str], optional): The clipdrop API key. Defaults
        to None.
        no_user_input (bool, optional): Don't ask for user input. Defaults to
        False.
        no_file_save (bool, optional): Don't save the files. Defaults to False.

    Returns:
        List[FullMeme]: The list of the memes.
        Its length may be less than `meme_count` if some memes were skipped due
        to errors.
    """
    # Display Header
    termcolor.cprint(
        f" AI Meme Generator - {__version__} ".center(
            shutil.get_terminal_size().columns, "="
        ),
        "blue",
    )
    print(
        """
AI Meme Generator
Originally created by ThioJoe <github.com/ThioJoe/Full-Stack-AI-Meme-Generator>
Modified by Koviubi56 in 2023.

Copyright (C) 2023  Koviubi56
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.
"""
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check if any settings arguments, and replace the default values with the
    # args if so. To run automated from command line, specify at least 1
    # argument.
    if (
        hasattr(args, "text_generation_service")
        and args.text_generation_service
    ):
        text_generation_service = args.text_generation_service
        termcolor.cprint(
            f"Text generation service will be {text_generation_service} from"
            " cli arguments",
            "cyan",
        )
    if hasattr(args, "text_model") and args.text_model:
        text_model = args.text_model
        termcolor.cprint(
            f"Text model will be {text_model} from cli arguments",
            "cyan",
        )
    if hasattr(args, "image_platform") and args.image_platform:
        image_platform = args.image_platform
        termcolor.cprint(
            f"Image platform will be {image_platform} from cli arguments",
            "cyan",
        )
    if hasattr(args, "temperature") and args.temperature:
        temperature = float(args.temperature)
        termcolor.cprint(
            f"Temperature will be {temperature} from cli arguments",
            "cyan",
        )
    if hasattr(args, "basic_instructions") and args.basic_instructions:
        basic_instructions = args.basic_instructions
        termcolor.cprint(
            f"Basic instructions will be {basic_instructions} from cli"
            " arguments",
            "cyan",
        )
    if (
        hasattr(args, "image_special_instructions")
        and args.image_special_instructions
    ):
        image_special_instructions = args.image_special_instructions
        termcolor.cprint(
            f"Image special instructions will be {image_special_instructions}"
            " from cli arguments",
            "cyan",
        )
    if hasattr(args, "no_file_save") and args.no_file_save:
        no_file_save = True
        termcolor.cprint(
            "Won't save file according to cli arguments",
            "cyan",
        )
    if hasattr(args, "no_user_input") and args.no_user_input:
        no_user_input = True
        termcolor.cprint(
            "Won't ask for user input according to cli arguments",
            "cyan",
        )
    settings = get_settings(no_user_input)
    use_config = settings.get("advanced", {}).get("use_this_config", True)
    if use_config:
        termcolor.cprint("Getting settings from config file...", "cyan")
        text_generation_service = settings.get("ai_settings", {}).get(
            "text_generation_service", text_generation_service
        )
        text_model = settings.get("ai_settings", {}).get(
            "text_model", text_model
        )
        temperature = float(  # float() is not necessary, but why not
            settings.get("ai_settings", {}).get("temperature", temperature)
        )
        basic_instructions = settings.get("basic", {}).get(
            "basic_instructions", basic_instructions
        )
        image_special_instructions = settings.get("basic", {}).get(
            "image_special_instructions", image_special_instructions
        )
        image_platform = settings.get("ai_settings", {}).get(
            "image_platform", image_platform
        )
        font_file_name = settings.get("advanced", {}).get(
            "font_file", font_file_name
        )
        base_file_name = settings.get("advanced", {}).get(
            "base_file_name", base_file_name
        )
        output_directory = settings.get("advanced", {}).get(
            "output_directory", output_directory
        )
    elif pathlib.Path(SETTINGS_FILE_NAME).exists():
        termcolor.cprint(
            "WARNING: The config file"
            f" {pathlib.Path(SETTINGS_FILE_NAME).resolve()} is ignored"
            " because `use_this_config` is not set to `true`!",
            "yellow",
        )

    output_directory = pathlib.Path(output_directory).resolve()

    # If API Keys not provided as parameters, get them from config file or
    # command line arguments
    if openai_key:
        termcolor.cprint("Getting API keys from function arguments...", "cyan")
        api_keys = APIKeys(openai_key, clipdrop_key, stability_key)
    else:
        api_keys = get_api_keys(args, no_user_input)

    if not no_user_input:
        # If no user prompt argument set, get user input for prompt
        if hasattr(args, "user_prompt") and args.user_prompt:
            user_entered_prompt = args.user_prompt
        else:
            print(
                "Enter a meme subject or concept (Or just hit enter to let"
                " the AI decide)"
            )
            user_entered_prompt = input(" >  ")
            if (
                not user_entered_prompt
            ):  # If user puts in nothing, set to "anything"
                user_entered_prompt = "anything"

        # If no meme count argument set, get user input for meme count
        if hasattr(args, "meme_count") and args.meme_count:
            meme_count = int(args.meme_count)
        else:
            # Set the number of memes to create
            meme_count = 1  # Default will be none if nothing entered
            print(
                "Enter the number of memes to create (Or just hit Enter for"
                " 1): "
            )
            user_entered_count = input(" >  ")
            if user_entered_count:
                meme_count = int(user_entered_count)

    # Get full path of font file from font file name
    font_file = check_font(font_file_name)

    def single_meme_generation_loop() -> Optional[FullMeme]:
        # Send request to chat bot to generate meme text and image prompt
        chat_response = text_generation_request(
            api_keys=api_keys,
            user_entered_prompt=user_entered_prompt,
            basic_instructions=basic_instructions,
            image_special_instructions=image_special_instructions,
            platform=text_generation_service,
            text_model=text_model,
            temperature=temperature,
        )

        # Take chat message and convert to dictionary with meme_text and
        # image_prompt
        meme = parse_meme(chat_response)
        if not meme:
            termcolor.cprint(
                "  Could not interpret response! Skipping", "yellow"
            )
            return None
        image_prompt = meme.image_prompt
        meme_text = meme.meme_text

        # Print the meme text and image prompt
        termcolor.cprint("  Meme Text:\t" + meme_text, "cyan")
        termcolor.cprint("  Image Prompt:\t" + image_prompt, "cyan")

        # Send image prompt to image generator and get image back
        # (Using DALL·E API)
        termcolor.cprint("  Sending image creation request...", "cyan")
        virtual_image_file = image_generation_request(
            api_keys, image_prompt, image_platform
        )

        # Combine the meme text and image into a meme
        file = set_file_path(base_file_name, output_directory)
        termcolor.cprint("  Creating full meme...", "cyan")
        virtual_meme_file = create_meme(
            virtual_image_file,
            meme_text,
            file,
            no_file_save=no_file_save,
            font_file=font_file,
        )

        termcolor.cprint("  Done!", "cyan")
        return FullMeme(meme_text, image_prompt, virtual_meme_file, file)

    # Create list of dictionaries to hold the results of each meme so that they
    # can be returned by main() if called from command line
    meme_results_dicts_list: List[FullMeme] = []

    for number in range(1, meme_count + 1):
        while True:
            termcolor.cprint("-" * shutil.get_terminal_size().columns, "blue")
            termcolor.cprint(
                f"Generating meme {number} of {meme_count}...", "cyan"
            )
            try:
                meme_info_dict = single_meme_generation_loop()
            except Exception:
                termcolor.cprint("Error while generating the meme:", "red")
                if no_user_input:
                    raise
                termcolor.cprint(traceback.format_exc(), "red")
                if no_user_input:
                    termcolor.cprint(
                        "WARNING: Skipping this meme...", "yellow"
                    )
                    break
                task = input("Skip, abort, retry? [S/a/r] ").lower()
                if task == "a":
                    sys.exit(1)
                if task == "r":
                    continue
                break

            # Add meme info dict to list of meme results
            if meme_info_dict:
                meme_results_dicts_list.append(meme_info_dict)
            break

    # If called from command line, will return the list of meme results
    return meme_results_dicts_list


if __name__ == "__main__":
    try:
        generate()
    except Exception:
        termcolor.cprint(traceback.format_exc(), "red")
        termcolor.cprint(
            "Please read the above error message! **If** you think this is a"
            " **bug**, please report it on GitHub including the **entire**"
            " above traceback.",
            "yellow",
        )
        try:
            input("Press Enter to exit...")
        except Exception:
            # If we can't use input (maybe we were ran in some automated system
            # that doesn't allow input) then just re-raise it
            raise
