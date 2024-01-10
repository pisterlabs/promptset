import os
import sys
import re
import argparse
from typing import Optional
import openai
import codecs


def parse_localization_string_file(file_path):
    localizations = {}

    with open(file_path, "r") as f:
        lines = f.readlines()
        current_language = None

        for line in lines:
            line = line.strip()
            if line.startswith("//"):
                current_language = line[3:].strip()
                localizations[current_language] = []
            elif current_language and "=" in line:
                localizations[current_language].append(line)

    return localizations


def generate_localization_instructions(
    template_file: str,
    languages_file: str,
    strings_file: str,
    extra_information: str = "",
):
    with open(template_file, "r") as f:
        template = f.read()

    # Assume that the languages file is a text file with one language per line, no empty lines,
    # and preceded by a *.
    with open(languages_file, "r") as f:
        languages = f.read()

    with open(strings_file, "r") as f:
        strings = f.read()

    formatted_template = template.format(
        languages=languages, extra_information=extra_information, strings=strings
    )
    return formatted_template


def copy_localizations_to_xcode_project(localizations, assets_folder):
    for language, strings in localizations.items():
        localizable_strings_file = os.path.join(
            assets_folder, f"{language}.lproj", "Localizable.strings"
        )

        with open(localizable_strings_file, "a") as f:
            for string in strings:
                f.write(string + "\n")


def main(
    input_file: str,
    assets_folder: str,
    openai_api_key: Optional[str],
    languages_file: str,
    template_file: str,
    extra_information: str,
    model: str,
    translations_output: str,
):
    openai.api_key = openai_api_key

    # Generate the localization instructions
    localization_instructions = generate_localization_instructions(
        template_file=template_file,
        languages_file=languages_file,
        strings_file=input_file,
        extra_information=extra_information,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who helps developers localize their apps. You are given a list of strings to localize. You are also given a list of languages to translate to. Listen to the instructions carefully and localize the strings in the way you're asked to.",
        },
        {"role": "user", "content": localization_instructions},
    ]

    response = {
        "choices": [
            {
                "finish_reason": None,
            }
        ]
    }

    translations = ""

    while response["choices"][0]["finish_reason"] != "stop":
        print("Prompting OpenAI to generate translations...")
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
        )
        print(f"Complete, finish reason: {response['choices'][0]['finish_reason']}")

        translations += response["choices"][0]["message"]["content"]

        messages.append(
            {
                "role": response["choices"][0]["message"]["role"],
                "content": response["choices"][0]["message"]["content"],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "continue translating",
            },
        )

    if translations == "":
        print("Error: No translations were generated.")
        sys.exit(1)

    print("\nTranslations generated.\n")

    # Write translations to a file in case we need to debug them
    # or just have a copy.
    print(f"Writing translations to {translations_output}...")
    with open(translations_output, "w") as f:
        f.write(translations)

    print(f"Successfully wrote translations to {translations_output}\n")

    # Parse the translations
    localizations = parse_localization_string_file(translations_output)

    print("Copying localization strings to the Xcode project...")
    copy_localizations_to_xcode_project(localizations, assets_folder)
    print("Successfully copied localization strings to the Xcode project.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy localization strings to the correct Xcode project files."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input file containing the strings to localize, in English (US).",
    )
    parser.add_argument(
        "assets_folder", help="Path to the assets folder in the Xcode project."
    )
    # Optional arguments
    parser.add_argument(
        "--openai-key",
        help="API key for OpenAI. If not provided, the script will use the key you provided before.",
    )
    parser.add_argument(
        "--languages-file",
        help="Path to the file containing the list of languages to translate to. Defaults to ./languages_list.txt.",
        default="languages_list.txt",
    )
    parser.add_argument(
        "--extra-information",
        help="Extra information to include in the instructions.",
        default="",
    )
    parser.add_argument(
        "--template-file",
        help="Path to the template file. Defaults to ./chatgpt_template.txt.",
        default="chatgpt_template.txt",
    )
    parser.add_argument(
        "--model",
        help="The OpenAI model to use. Defaults to gpt-4.",
        default="gpt-4",
    )
    parser.add_argument(
        "--translations-output",
        help="Path to the file where translations will be written. Defaults to ./translations.txt.",
        default="translations.txt",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: {args.input_file} is not a valid file")
        sys.exit(1)

    if not os.path.isdir(args.assets_folder):
        print(f"Error: {args.assets_folder} is not a valid directory")
        sys.exit(1)

    if not os.path.isfile(args.languages_file):
        print(f"Error: {args.languages_file} is not a valid file")
        sys.exit(1)

    if not os.path.isfile(args.template_file):
        print(f"Error: {args.template_file} is not a valid file")
        sys.exit(1)

    # Write the OpenAI API key to a file, .key
    if args.openai_key:
        with open(".key", "w") as f:
            f.write(args.openai_key)
            print("Successfully wrote the OpenAI API key to .key")

        openai_api_key = args.openai_key
    else:
        print("No OpenAI API key provided. Using the key you provided before.")
        with open(".key", "r") as f:
            openai_api_key = f.read()

    if not openai_api_key:
        print("Error: No OpenAI API key provided.")
        sys.exit(1)

    main(
        input_file=args.input_file,
        assets_folder=args.assets_folder,
        openai_api_key=openai_api_key,
        languages_file=args.languages_file,
        template_file=args.template_file,
        extra_information=args.extra_information,
        model=args.model,
        translations_output=args.translations_output,
    )
