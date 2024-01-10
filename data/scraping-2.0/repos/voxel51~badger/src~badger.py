"""
| Badger CLI.
| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import argparse
import os
import pyperclip
import sys
import yaml

HOME_DIR = os.path.expanduser("~")
DEFAULT_BADGER_DIR = os.path.join(HOME_DIR, ".badger")
BADGER_CONFIG_DIR = os.environ.get("BADGER_CONFIG_DIR", DEFAULT_BADGER_DIR)
BADGER_CONFIG_FILE = os.path.join(BADGER_CONFIG_DIR, "config.yaml")


KEYS = (
    "logo",
    "url",
    "text",
    "color",
    "style",
    "logoColor",
    "label",
    "labelColor",
    "logoWidth",
)

COMMON_COLORS = (
    "brightgreen",
    "green",
    "yellowgreen",
    "yellow",
    "orange",
    "red",
    "blue",
    "lightgrey",
    "success",
    "important",
    "critical",
    "informational",
    "inactive",
)

OPTIONAL_KEYS = (
    "text",
    "color",
    "style",
    "logoColor",
    "label",
    "labelColor",
    "logoWidth",
)

REQUIRED_KEYS = ("logo", "url", "badge_name")

CONTRIBUTOR_LABEL_COLOR = "212529"
CONTRIBUTOR_COLORS = ("FF6D04", "499CEF", "6D04FF", "59A65C")


from .utils import (
    list_badges,
    generate_badge_markdown,
    save_svg_file,
    save_as_badge,
    print_badge_info,
    extract_name_from_github,
    create_unique_badge_name,
)

from .go_wild_utils import generate_random_svg, generate_trial_badge


def get_required_string(key):
    if key in OPTIONAL_KEYS:
        return f"[OPTIONAL]"
    elif key in REQUIRED_KEYS:
        return f"[REQUIRED]"


class BadgerConfig:
    def __init__(self, config_file):
        self.config_file = config_file

        if not os.path.exists(self.config_file):
            create_default_config(self.config_file)
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, "r") as file:
            return yaml.safe_load(file).get("badges", {})

    def update_config(self, new_config):
        with open(self.config_file, "w") as file:
            yaml.safe_dump({"badges": new_config}, file)


# Function to create a default config file with a 'badger' badge
def create_default_config(file_path):
    # Create the directory if it does not exist
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build the path to badger.svg relative to the script's location
    badger_svg_path = os.path.join(script_dir, "../assets/badger.svg")

    default_badge = {
        "badger": {
            "url": "https://github.com/voxel51/badger",
            "color": "blue",
            "logo": badger_svg_path,
            "text": "Badger",
            "logoColor": "white",
            "style": "flat-square",
        }
    }

    with open(file_path, "w") as file:
        yaml.safe_dump({"badges": default_badge}, file)

    print(f"Created default config file at {file_path}")


def parse_badger_file():
    with open(BADGER_CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
        return config.get("badges", {})


def create_badge(args, config, simple=False):
    """
    Create a new badge based on the provided command-line arguments and update the configuration.
    """

    def _get_color():
        print("Commonly used colors:")
        for i, color in enumerate(COMMON_COLORS, 1):
            print(f"{i}. {color}")

        req = get_required_string("color")
        color_choice = input(
            f"{req} Enter the color of the badge (or choose a number from the list above): "
        )

        # If the user enters a number, map it to the corresponding color
        if color_choice.isdigit():
            color_choice = int(color_choice) - 1
            if 0 <= color_choice < len(COMMON_COLORS):
                color = COMMON_COLORS[color_choice]
            else:
                print("Invalid choice, using default color 'blue'.")
                color = "blue"
        else:
            color = color_choice

        return color

    badges = config.load_config()  # Load the current config

    SIMPLE_FLAG = simple

    if args.badge_name:
        badge_name = args.badge_name
    else:
        req = get_required_string("badge_name")
        if SIMPLE_FLAG:
            badge_name = create_unique_badge_name(badges)
            print(f"Using unique name '{badge_name}'.")
        else:
            badge_name = input(f"{req} Enter the name of the new badge: ")

    # Check for duplicate badge name
    while badge_name in badges:
        overwrite = input(
            f"A badge with the name '{badge_name}' already exists. Do you want to overwrite it? (y/n): "
        ).lower()
        if overwrite == "y":
            break
        elif overwrite == "n":
            badge_name = input("Enter a new name for the badge: ")
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # Dynamic badge data collection
    new_badge_data = {}

    for key in KEYS:
        if getattr(args, key, None):
            value = getattr(args, key)
        elif key == "color" and not SIMPLE_FLAG:
            value = _get_color()
        elif not SIMPLE_FLAG:
            req = get_required_string(key)
            value = input(f"{req} Enter the {key} for the badge: ")
        else:
            continue

        if value:  # Only add non-empty values to the config
            new_badge_data[key] = value

    # Update the badges dictionary and save it back to the config file
    if all(key in new_badge_data for key in ("logo", "url")):
        badges[badge_name] = new_badge_data
        config.update_config(badges)
        print(f"Successfully added badge '{badge_name}'.")
    else:
        print("Skipping badge creation due to missing required fields.")
        print(f"Required fields: {REQUIRED_KEYS}")


def create_fiftyone_contributor_badge(args, config, simple=False):
    """
    Create a contributor badge based on the provided command-line arguments and update the configuration.
    """

    CONTRIBUTOR_KEYS = ("username", "name", "variant", "style", "logoWidth")
    GH_URL = "https://github.com/voxel51/fiftyone/commits?author="

    SIMPLE_FLAG = simple

    badges = config.load_config()  # Load the current config

    if args.badge_name:
        badge_name = args.badge_name
    else:
        req = get_required_string("badge_name")
        if SIMPLE_FLAG:
            badge_name = create_unique_badge_name(badges)
            print(f"Using unique name '{badge_name}'.")
        else:
            badge_name = input(f"{req} Enter the name of the new badge: ")

    # Check for duplicate badge name
    while badge_name in badges:
        overwrite = input(
            f"A badge with the name '{badge_name}' already exists. Do you want to overwrite it? (y/n): "
        ).lower()
        if overwrite == "y":
            break
        elif overwrite == "n":
            badge_name = input("Enter a new name for the badge: ")
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # Dynamic badge data collection
    new_badge_data = {}
    new_badge_data["label"] = "contributor"
    new_badge_data["labelColor"] = CONTRIBUTOR_LABEL_COLOR
    new_badge_data["logo"] = "assets/fiftyone.svg"
    new_badge_data["logoColor"] = "white"

    def _variant_to_color(variant):
        if not variant.isdigit():
            variant = 1
        ## typecast to int and subtract 1 to get the correct index
        variant = int(variant) - 1
        return "color", CONTRIBUTOR_COLORS[variant]

    def _username_to_url(username):
        return "url", GH_URL + username

    def _name_to_text(name):
        return "text", name

    def _handle_props(value, key):
        if key == "variant":
            return _variant_to_color(value)
        elif key == "username":
            return _username_to_url(value)
        elif key == "name":
            return _name_to_text(value)
        else:
            return key, value

    for key in CONTRIBUTOR_KEYS:
        if getattr(args, key, None):
            value = getattr(args, key)
            key, value = _handle_props(value, key)
        elif key == "username":
            value = input(f"Enter the GitHub username to link to: ")
            key, value = _handle_props(value, key)
        elif key == "name":
            username = new_badge_data["url"].split("=")[-1]
            if username:
                value = extract_name_from_github(username)
                if not value and not SIMPLE_FLAG:
                    value = input(
                        f"Enter the name to be displayed on the badge: "
                    )
            elif not SIMPLE_FLAG:
                value = input(f"Enter the name to be displayed on the badge: ")
            key, value = _handle_props(value, key)
        elif key == "variant" and not SIMPLE_FLAG:
            value = input(f"Enter the variant of the badge [1-4]: ")
            key, value = _handle_props(value, key)
        else:
            req = get_required_string(key)
            if not SIMPLE_FLAG:
                value = input(f"{req} Enter the {key} for the badge: ")
                key, value = _handle_props(value, key)
            else:
                continue
        new_badge_data[key] = value

    # Update the badges dictionary and save it back to the config file
    badges[badge_name] = new_badge_data
    config.update_config(badges)

    print(f"Successfully added badge '{badge_name}'.")


def delete_badge(args, config):
    """
    Delete a badge based on the provided command-line arguments and update the configuration.
    """
    badges = config.load_config()  # Load the current config

    badge_name = args.badge_name

    # Check if the badge exists
    if badge_name not in badges:
        print(f"No badge found with the name '{badge_name}'.")
        return

    # Confirmation prompt
    confirmation = input(
        f"Are you sure you want to delete the badge '{badge_name}'? (y/n): "
    ).lower()

    if confirmation == "y":
        # Delete the badge and update the config file
        del badges[badge_name]
        config.update_config(badges)
        print(f"Successfully deleted badge '{badge_name}'.")
    elif confirmation == "n":
        print("Operation cancelled.")
    else:
        print("Invalid input. Operation cancelled.")


def copy_or_print_badge(args, config, action="copy"):
    """
    Copy or print badge Markdown based on the provided command-line arguments and current configuration.
    :param args: Command-line arguments
    :param config: An instance of the BadgerConfig class
    :param action: Either "copy" to copy the badge Markdown to clipboard or "print" to print it to the terminal
    """
    badges = config.load_config()  # Load the current config

    badge_name = args.badge_name

    # Check if the badge exists
    if badge_name not in badges:
        print(f"No badge found with the name '{badge_name}'.")
        return

    badge_config = badges[badge_name]

    # Override the defaults with any provided command-line arguments
    for key in KEYS:
        if getattr(args, key, None):
            badge_config[key] = getattr(args, key)

    badge_markdown = generate_badge_markdown(badge_config)

    if action == "copy":
        # Copy to clipboard
        pyperclip.copy(badge_markdown)
        print(f"Copied badge '{badge_name}' to clipboard.")
    elif action == "print":
        # Print to terminal
        print(badge_markdown)
    else:
        print("Invalid action specified. Please use 'copy' or 'print'.")


def clone_badge(args, config):
    """
    Clone an existing badge to a new name based on the provided command-line arguments and update the configuration.
    """
    badges = config.load_config()  # Load the current config

    original_badge_name = args.badge_name
    new_badge_name = args.new_badge_name

    # Check if the original badge exists
    if original_badge_name not in badges:
        print(f"No badge found with the name '{original_badge_name}'.")
        return

    # Check if the new badge name already exists
    if new_badge_name in badges:
        print(f"A badge with the name '{new_badge_name}' already exists.")
        return

    # Clone the badge
    original_badge_data = badges[original_badge_name]
    new_badge_data = original_badge_data.copy()

    # Override the defaults with any provided command-line arguments
    for key in KEYS:
        if getattr(args, key, None):
            new_badge_data[key] = getattr(args, key)

    badges[new_badge_name] = new_badge_data

    # Update the config file
    config.update_config(badges)

    print(
        f"Successfully cloned badge '{original_badge_name}' to '{new_badge_name}'."
    )


def edit_badge(args, config):
    """
    Edit an existing badge based on the provided command-line arguments and update the configuration.
    """
    badges = config.load_config()  # Load the current config

    badge_name = args.badge_name

    # Check if the badge exists
    if badge_name not in badges:
        print(f"No badge found with the name '{badge_name}'.")
        return

    badge_config = badges[badge_name]

    # Show current settings
    print(f"Current settings for '{badge_name}':")
    for key, value in badge_config.items():
        print(f"{key}: {value}")

    print("Performing edit...")
    # Edit settings
    for key in KEYS:
        new_value = getattr(args, key, None)
        if new_value is not None:
            badge_config[key] = new_value
            print(f"--> Updated {key} to {new_value}.")

    # Update the badge and config file
    badges[badge_name] = badge_config
    config.update_config(badges)

    print(f"Successfully updated badge '{badge_name}'.")


def go_wild(args, config):
    try:
        import openai
    except ImportError:
        print(
            "The openai package is not installed. Please install it by running 'pip install openai'"
        )
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    svg_data = generate_random_svg(args.prompt)
    if not svg_data:
        return

    # Copy to clipboard
    pyperclip.copy(generate_trial_badge(svg_data, args.prompt))
    print("SVG code has been copied to your clipboard.")

    save_option = input("Do you want to save this SVG? (y/n): ")
    if save_option.lower() == "y":
        file_name = save_svg_file(svg_data)
        save_as_badge(file_name, config)


def get_badge_info(args, config):
    badge_name = args.badge_name
    print_badge_info(badge_name, config.load_config())


def main():
    SIMPLE_FLAG = "--simple" in sys.argv
    # Remove --simple to avoid conflicts
    while "--simple" in sys.argv:
        sys.argv.remove("--simple")

    parser = argparse.ArgumentParser(
        description="Manage custom badges for GitHub READMEs."
    )
    # parser.add_argument(
    #     "--simple", action="store_true", help="Skip interactive inputs."
    # )
    subparsers = parser.add_subparsers(dest="command")

    ### CREATE
    create_parser = subparsers.add_parser("create", help="Create a new badge.")
    create_parser.add_argument(
        "badge_name", nargs="?", default=None, help="Name of the new badge."
    )
    # create_parser.add_argument(
    #     "--simple", action="store_true", help="Skip interactive inputs."
    # )

    ### DELETE
    delete_parser = subparsers.add_parser("delete", help="Delete a badge.")
    delete_parser.add_argument(
        "badge_name", help="Name of the badge to delete."
    )

    ### EDIT
    edit_parser = subparsers.add_parser("edit", help="Edit a badge.")
    edit_parser.add_argument("badge_name", help="Name of the badge to edit.")

    ### COPY
    copy_parser = subparsers.add_parser(
        "copy", help="Copy badge Markdown to clipboard."
    )
    copy_parser.add_argument("badge_name", help="Name of the badge to copy.")

    ### PRINT
    print_parser = subparsers.add_parser(
        "print", help="Print badge Markdown to terminal."
    )
    print_parser.add_argument("badge_name", help="Name of the badge to print.")

    ### CLONE
    clone_parser = subparsers.add_parser("clone", help="Clone a badge.")
    clone_parser.add_argument("badge_name", help="Name of the badge to clone.")
    clone_parser.add_argument("new_badge_name", help="Name of the new badge.")

    for sp in [
        create_parser,
        copy_parser,
        print_parser,
        clone_parser,
        edit_parser,
    ]:
        sp.add_argument("--text", help="Text override.")
        sp.add_argument("--color", help="Color override.")
        sp.add_argument("--logo", help="Logo file override.")
        sp.add_argument("--url", help="URL override.")
        sp.add_argument("--style", help="Style override.")
        sp.add_argument("--logoColor", help="Logo color override.")
        sp.add_argument("--label", help="Label override.")
        sp.add_argument("--labelColor", help="Label color override.")
        sp.add_argument("--logoWidth", help="Logo width override.")

    ### CREATE FIFTYONE CONTRIBUTOR BADGE
    contributor_parser = subparsers.add_parser(
        "contribute", help="Create a new fiftyone contributor badge."
    )
    contributor_parser.add_argument(
        "--badge_name", help="Name of the new badge."
    )
    contributor_parser.add_argument(
        "--name", help="Name to be displayed on the badge."
    )
    contributor_parser.add_argument(
        "--username", help="GitHub username to link to."
    )
    contributor_parser.add_argument(
        "--variant", help="Variant of the badge [1-4]."
    )
    contributor_parser.add_argument("--style", help="Style override.")
    contributor_parser.add_argument("--logoWidth", help="Logo width override.")
    # contributor_parser.add_argument(
    #     "--simple", action="store_true", help="Skip interactive inputs."
    # )

    ### GO WILD
    go_wild_parser = subparsers.add_parser(
        "go-wild", help="Generate a random SVG badge."
    )
    go_wild_parser.add_argument(
        "--prompt", help="Subject of the SVG to generate.", default="badger"
    )

    ### INFO
    info_parser = subparsers.add_parser("info", help="Get info about a badge.")
    info_parser.add_argument("badge_name", help="Name of the badge.")

    ### LIST BADGES AND HELP
    subparsers.add_parser("list", help="List available badges.")
    subparsers.add_parser("help", help="List available commands.")

    args = parser.parse_args()

    # Initialize configuration
    badger_config = BadgerConfig(BADGER_CONFIG_FILE)
    badges = badger_config.load_config()

    if args.command == "create":
        create_badge(args, badger_config, simple=SIMPLE_FLAG)
    elif args.command == "contribute":
        create_fiftyone_contributor_badge(
            args, badger_config, simple=SIMPLE_FLAG
        )
    elif args.command == "delete":
        delete_badge(args, badger_config)
    elif args.command == "copy":
        copy_or_print_badge(args, badger_config, action="copy")
    elif args.command == "print":
        copy_or_print_badge(args, badger_config, action="print")
    elif args.command == "clone":
        clone_badge(args, badger_config)
    elif args.command == "edit":
        edit_badge(args, badger_config)
    elif args.command == "go-wild":
        go_wild(args, badger_config)
    elif args.command == "info":
        get_badge_info(args, badger_config)
    elif args.command == "list":
        list_badges(badges)
    elif args.command == "help":
        parser.print_help()


if __name__ == "__main__":
    main()
