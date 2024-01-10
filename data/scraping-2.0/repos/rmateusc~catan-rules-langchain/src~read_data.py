import argparse
import os
import re
from pathlib import Path
from typing import Dict, Union

from langchain.document_loaders import PyPDFLoader

SRC_DIR = Path(__file__).parents[1]
DATA_DIR = SRC_DIR / "data"


# setup argparse
def parse_args() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reload_pdfs",
        type=bool,
        default=False,
        help="Reload the pdf files and save them as text files.",
    )
    args = parser.parse_args()

    return vars(args)


def read_raw_pdfs(folder_to_save: str) -> None:
    """
    Read the raw pdf files and save them as text files in data/text_files folder.
    """
    # explore the raw data folder
    for file in os.listdir(DATA_DIR / "raw"):
        loader = PyPDFLoader(str(DATA_DIR / "raw" / file))
        pages = loader.load_and_split()
        # store all pages in a single string
        text = ""
        for page in pages:
            text += page.page_content.replace("•", "").replace("\u200aY", "")
            # delete everything after Credits
            text = text.split("Credits")[0]
            text = text.split("index")[0]

        # add title of Game Rules
        title = "Catan"
        if "base" in file.lower():
            title += " Base Game"
        elif "seafarers" in file.lower():
            title += " Seafarers"
        elif "cities" in file.lower():
            title += " Cities & Knights"
        # add 5-6 players diffentiation
        if "expansion" in file.lower():
            title += " Expansion 5 & 6 players"
        title += " Game Rules: \n "
        # append the title to the text
        text = title.upper() + text

        # remove irregulat characters
        patterns_to_remove = [
            r"\n\d+",  # Remove '\n' followed by numbers
            r"\n\s*[A-Z]",
            # r'\n'  # Remove '\n' followed by any capital letter
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text)

        # see if data/processed folder exists
        if not os.path.exists(DATA_DIR / folder_to_save):
            os.makedirs(DATA_DIR / folder_to_save)

        # save the text files
        with open(DATA_DIR / folder_to_save / file.replace(".pdf", ".txt"), "w") as f:
            f.write(text)


def main(reload_pdfs: bool):
    folder_to_save = "text_files"
    if reload_pdfs:
        # if the folder exists, delete it
        if os.path.exists(DATA_DIR / folder_to_save):
            # delete whole folder with all files
            for file in os.listdir(DATA_DIR / folder_to_save):
                os.remove(DATA_DIR / folder_to_save / file)
            os.rmdir(DATA_DIR / folder_to_save)

    if os.path.exists(DATA_DIR / folder_to_save):
        print("The folder already exists")
    else:
        print("Loading the pdf files...")
        read_raw_pdfs(folder_to_save)


if __name__ == "__main__":
    args = parse_args()
    main(**args)
