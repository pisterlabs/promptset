import logging
import subprocess
import argparse

from pathlib import Path
from .utility_fns import construct_dirty_split, parse_zerogpt_result, parse_openai_result

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path.cwd()
CACHE_PATH = Path(PROJECT_ROOT, "result", "cache")
DATA_PATH = Path(PROJECT_ROOT, "data")

parser = argparse.ArgumentParser(
    description="Dataset downloader.\nUse the command line argument to select which version of OpenGPTText and (pre-processed) OpenWebText you want to download"
)
parser.add_argument(
    "--original",
    action="store_true",
    help="Download the raw version of OpenGPTText and OpenWebText"
)
parser.add_argument(
    "--strip",
    action="store_true",
    help="Download the stripped (remove consequtive newlines) version of OpenGPTText and OpenWebText"
)
parser.add_argument(
    "--ascii",
    action="store_true",
    help="Download the normalized (cast Unicode to ASCII character) version of OpenGPTText and OpenWebText"
)
parser.add_argument(
    "--final",
    action="store_true",
    help="Download the OpenGPTText-Final and sanitized version of OpenWebText (final = ascii |> strip)"
)
parser.add_argument(
    "--split",
    action="store_true",
    help="Download the splitted version of Datasets (train, valid, test). Used for training and evaluation"
)
parser.add_argument(
    "--baseline",
    action="store_true",
    help="Download the raw response from OpenAI classifier and ZeroGPT classifier on all three test datasets"
)


def download_original():
    gpt_link = "https://drive.google.com/drive/folders/1VpTAgEGMmPSaS1yaMuN1JLufG-N1UPYo"
    web_link = "https://drive.google.com/drive/folders/1VpPe0-JChp25Jj-WHgqQ7sWtzXKnty1T"
    logger.info("Download open-gpt-text...")
    subprocess.run(["gdown", "--folder", gpt_link], cwd="data")
    logger.info("Downloading open-web-text...")
    subprocess.run(["gdown", "--folder", web_link], cwd="data")


def download_strip():
    gpt_strip_link = "https://drive.google.com/drive/folders/1WN2g3AGDU8t7ZQN2st4BNfqi7qvBiwXB"
    web_strip_link = "https://drive.google.com/drive/folders/1WgP4y6RVlNmElqngpnuwlIJguZ3ChrUQ"
    logger.info("Downloading open-gpt-text-strip...")
    subprocess.run(["gdown", "--folder", gpt_strip_link], cwd="data")
    logger.info("Downloading open-web-text-strip...")
    subprocess.run(["gdown", "--folder", web_strip_link], cwd="data")


def download_ascii():
    gpt_ascii_link = "https://drive.google.com/drive/folders/1Bg8hMgrU_w8GUL3VosSCvyDkuZ3ziz-P"
    web_ascii_link = "https://drive.google.com/drive/folders/1knPCQRI7pw8Y0DCYD2zT09UjA9CVrRlB"
    logger.info("Downloading open-gpt-text-ascii...")
    subprocess.run(["gdown", "--folder", gpt_ascii_link], cwd="data")
    logger.info("Downloading open-web-text-ascii...")
    subprocess.run(["gdown", "--folder", web_ascii_link], cwd="data")


def download_final():
    gpt_final_link = "https://drive.google.com/drive/folders/1uF3c3Cx3-6A6gbCkkXdvnw9-yZE7lAh9"
    web_final_link = "https://drive.google.com/drive/folders/1uHaxvbai6MshHfmM4NpOsqO2_itS_V5e"
    logger.info("Downloading open-gpt-text-final...")
    subprocess.run(["gdown", "--folder", gpt_final_link], cwd="data")
    logger.info("Downloading open-web-text-final...")
    subprocess.run(["gdown", "--folder", web_final_link], cwd="data")


def download_split():
    gpt_split_link = "https://drive.google.com/drive/folders/1uc9kB5Nm7zx1UMJcpeGnjU7GI6xWNxNy"
    web_split_link = "https://drive.google.com/drive/folders/1raU6ST48KziMc6C8Kt8wcufJWPH4HKUo"
    logger.info("Downloading open-gpt-text-split...")
    subprocess.run(["gdown", "--folder", gpt_split_link], cwd="data")
    logger.info("Downloading open-web-text-split...")
    subprocess.run(["gdown", "--folder", web_split_link], cwd="data")
    # We need to construct the dirty split test dataset manually
    print("Constructing test-dirty.jsonl")
    construct_dirty_split()


def download_baseline():
    openai_link = "https://drive.google.com/drive/folders/15x4PjzIb2J64c9zGI9HIpea0ESd72duf"
    zerogpt_link = "https://drive.google.com/drive/folders/1gDXiyvSuuZAAMsQM0EotNT5ckn4f6D9B"
    logger.info("Downloading openai_classifier_output...")
    subprocess.run(["gdown", "--folder", openai_link], cwd="data")
    logger.info("Downloading zerogpt_classifier_output...")
    subprocess.run(["gdown", "--folder", zerogpt_link], cwd="data")

    # Parse raw response
    print("Parsing raw response from ZeroGPT")
    zerogpt_path = Path(DATA_PATH, "zerogpt_classifier_output")
    zerogpt_tasks = [
        # [web_response, gpt_response, output_path]
        [
            Path(zerogpt_path, "web-zerogpt.jsonl"), 
            Path(zerogpt_path, "gpt-zerogpt.jsonl"),
            Path(CACHE_PATH, "eval_zerogpt_opengpt_final.pt")
        ],
        [
            Path(zerogpt_path, "web-dirty-zerogpt.jsonl"),
            Path(zerogpt_path, "gpt-dirty-zerogpt.jsonl"),
            Path(CACHE_PATH, "eval_zerogpt_opengpt_original.pt")
        ],
        [
            Path(zerogpt_path, "gpt2-output-web-zerogpt.jsonl"),
            Path(zerogpt_path, "gpt2-output-gpt-zerogpt.jsonl"),
            Path(CACHE_PATH, "eval_zerogpt_gpt2_output.pt")
        ]
    ]
    for task in zerogpt_tasks: parse_zerogpt_result(*task)

    print("Parsing raw response from OpenAI")
    openai_path = Path(DATA_PATH, "openai_classifier_output")
    openai_tasks = [
        # [web_response, gpt_response, output_path]
        [
            Path(openai_path, "web-openai.jsonl"), 
            Path(openai_path, "gpt-openai.jsonl"),
            Path(CACHE_PATH, "eval_openai_opengpt_final.pt")
        ],
        [
            Path(openai_path, "web-dirty-openai.jsonl"),
            Path(openai_path, "gpt-dirty-openai.jsonl"),
            Path(CACHE_PATH, "eval_openai_opengpt_original.pt")
        ],
        [
            Path(openai_path, "gpt2-output-web-openai.jsonl"),
            Path(openai_path, "gpt2-output-gpt-openai.jsonl"),
            Path(CACHE_PATH, "eval_openai_gpt2_output.pt")
        ]
    ]
    for task in openai_tasks: parse_openai_result(*task)


if __name__ == "__main__":
    args = vars(parser.parse_args())

    download_tasks = []
    available_actions = [download_original, download_strip,
                         download_ascii, download_final, download_split, download_baseline]
    available_options = ["original", "strip",
                         "ascii", "final", "split", "baseline"]

    action_count = 0

    print("Subset to download:", end=" ")
    for argument in available_options:
        if args[argument]:
            print(argument, end=", ")
            action_count += 1
    print("\n")
    if action_count == 0:
        print("You need to select at least one subset to download. Abort.\n----------\n")
        parser.print_help()
        exit()

    confirm = input("Press y/Y to continue: ")
    if confirm.lower() != "y":
        print("Download abort.\n----------\n")
        parser.print_help()
        exit()

    for argument, action in zip(available_options, available_actions):
        if args[argument]:
            print("\n\n\n Downloading", argument, "\n\n ----------\n\n")
            action()
