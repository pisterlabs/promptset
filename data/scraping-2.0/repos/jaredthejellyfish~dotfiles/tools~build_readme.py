import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from pprint import pprint

print = pprint

load_dotenv()

client = OpenAI()

keymap = {
    "0x29": "ñ",
    "0x2F": ".",
    "0x2B": ",",
    "0x2A": "ç",
    "0x27": "´",
    "0x14": "3",
}


def generate_keybinds_section(keybinds: dict):
    prompt = "\n".join(keybinds)

    prompt = f"## Keybinds\n\n{prompt}\n\n"

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that is trained to rewrite pieces of text. You only respond with the answer you were asked for. You provide no context or do anything you are not directly told to do. You are extremely good at categorizing, you are very granular and detailed in your categorizations. ",
            },
            {
                "role": "user",
                "content": f"I am going to give you a piece of markdown text with all the keybinds running on my computer at the moment, I want you to split the keybinds into categories and subcategories so I can then place that as a section into my README.md file.\n\n{prompt}",
            },
        ],
        temperature=0,
        max_tokens=3050,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response.choices[0].message.content

    # with open(os.path.expanduser("~/.config/README.md"), "r") as f:
    #     lines = f.readlines()

    # keybinds_section = generate_keybinds_section(keybinds_list)
    # with open(os.path.expanduser("~/.config/README.md"), "w") as f:
    #     f.writelines(lines_start)


# Open the skhdrc file
with open(os.path.expanduser("~/.config/skhd/skhdrc"), "r") as f:
    skhd_lines = f.readlines()
    # Remove empty lines and comments
    skhd_lines = [
        {
            "keybind": re.split(r"\s*[+-]\s*", line.split(":")[0]),
            "action": line.split("#")[-1].strip(),
        }
        for line in skhd_lines
        if line.strip() and not line.startswith("#")
    ]

    # Replace key names with their corresponding values from the keymap
    for entry in skhd_lines:
        for i, key in enumerate(entry["keybind"]):
            if key in keymap:
                entry["keybind"][i] = keymap[key]

    # Sort the keybinds by their first key
    skhd_lines = sorted(skhd_lines, key=lambda x: x["keybind"][0])

    formatted_keybinds = []
    for bind in skhd_lines:
        keys = " + ".join(bind["keybind"])
        formatted_keybinds.append(
            "<kbd>{}</kbd> : {}<br />".format(keys, bind["action"]))

    keybinds = generate_keybinds_section(formatted_keybinds)

    with open(os.path.expanduser("~/.config/README.md"), "r") as f:
        readme_lines = f.readlines()

        # find the first occurence of "## Keybinds"
        keybinds_start = readme_lines.index("## Keybinds\n")

        # find the last occurence of "</kbd> :"

        keybinds_end = [
            i
            for i, line in enumerate(readme_lines)
            if "</kbd> :" in line and i > keybinds_start
        ][-1]

        # replace the keybinds section with the new keybinds
        readme_lines = (
            readme_lines[: keybinds_start]
            + [keybinds]
            + readme_lines[keybinds_end + 1:]
        )

        with open(os.path.expanduser("~/.config/README.md"), "w") as f:
            f.writelines(readme_lines)

            exit(0)
