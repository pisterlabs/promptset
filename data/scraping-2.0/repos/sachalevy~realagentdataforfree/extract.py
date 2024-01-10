from pathlib import Path
import os
import json
from typing import List
import argparse

import openai
import datasets
import requests
from dotenv import load_dotenv

import prompt, utils

load_dotenv(".env")

client = openai.OpenAI()
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
}


def get_api_signature_from_vision(filepath: Path, user_prompt: str):
    """
    Ask gpt-4v to generate an api signature describing the user's actions
    based on the provided screenshot.
    """
    image_payload = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{utils.encode_image(filepath)}"},
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": prompt.SYSTEM_PROMPT_API_SIGNATURE_FROM_VISION,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    image_payload,
                ],
            },
        ],
        "max_tokens": 512,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        return response.json(), response.json().get("choices")[0].get("message").get(
            "content"
        )
    except Exception as e:
        return {}, None


def get_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-3.5-turbo-1106",
    request_json: bool = False,
):
    completion_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    if request_json:
        completion_kwargs["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**completion_kwargs)

    return completion, completion.choices[0].message.content


def main(screenshot_dir: Path, samples_filepath: Path, use_vision: bool = False):
    assert screenshot_dir.exists(), "Screenshot directory does not exist."
    screenshot_filenames = [x.stem for x in screenshot_dir.iterdir()]
    screenshot_filenames = sorted(
        screenshot_filenames, key=lambda x: int(x.split("_")[1])
    )
    screenshot_timestamps = [int(x.split("_")[1]) for x in screenshot_filenames]

    keystroke_filepath = Path("data/presses.txt")
    scrolls_filepath = Path("data/scrolls.txt")
    clicks_filepath = Path("data/clicks.txt")

    output_complete_samples_fd = open(samples_filepath, "a")
    samples = []
    for i in range(len(screenshot_filenames) - 1):
        screenshot_filename = screenshot_filenames[i]
        start_ts, end_ts = screenshot_timestamps[i], screenshot_timestamps[i + 1]

        # get all events happening between start_ts and end_ts
        keystrokes = utils.retrieve_current_event(keystroke_filepath, start_ts, end_ts)
        scrolls = utils.retrieve_current_event(scrolls_filepath, start_ts, end_ts)
        clicks, active_app = utils.retrieve_current_event(
            clicks_filepath, start_ts, end_ts
        )

        # extract user entries + on-screen text context
        sentences = utils.extract_sentences_from_keystrokes(keystrokes)
        screenshot_filepath = screenshot_dir / (screenshot_filename + ".png")
        text_context = utils.extract_text_from_screenshot(screenshot_filepath)
        action_description_user_prompt = prompt.USER_PROMPT_ACTION_DESCRIPTION.format(
            displayed_text=text_context,
            user_input=" ".join(sentences),
            active_app=active_app,
        )
        _, action_description = get_completion(
            prompt.SYSTEM_PROMPT_ACTION_DESCRIPTION,
            action_description_user_prompt,
        )

        api_signature_user_prompt = prompt.USER_PROMPT_API_SIGNATURE.format(
            action_description=action_description,
            user_input=" ".join(sentences),
            window_context=text_context,
            active_app=active_app,
        )
        if use_vision:
            _, api_signature = get_api_signature_from_vision(
                screenshot_filepath, api_signature_user_prompt
            )
        else:
            _, api_signature = get_completion(
                prompt.SYSTEM_PROMPT_API_SIGNATURE_FROM_TEXT, api_signature_user_prompt
            )

        inferred_api_call_arguments_user_prompt = prompt.INFER_ARGS_USER_PROMPT.format(
            action_description=action_description,
            user_input=" ".join(sentences),
            window_context=text_context,
            active_app=active_app,
            api_signature=api_signature,
        )
        _, inferred_args = get_completion(
            prompt.INFER_ARGS_SYSTEM_PROMPT,
            inferred_api_call_arguments_user_prompt,
            request_json=True,
        )
        inferred_args = json.loads(inferred_args)

        sample = {
            "screenshot_filepath": str(screenshot_filepath),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration": end_ts - start_ts,
            "active_app": active_app,
            "action_description": action_description,
            "api_signature": json.dumps(api_signature)
            if isinstance(api_signature, dict)
            else api_signature,
            "user_input": " ".join(sentences),
            "inferred_args": inferred_args,
            "keystrokes": keystrokes,
            "scrolls": scrolls,
            "clicks": clicks,
        }
        output_complete_samples_fd.write(json.dumps(sample) + "\n")
        samples.append(sample)

    return samples


def compile_arrow_dataset(samples: List[dict], output_dir: Path):
    formatted_samples = []
    for sample in samples:
        formatted_sample = prompt.SAMPLE_PROMPT.format(
            api_signature=sample["api_signature"],
            user_prompt=sample["inferred_args"]["user_prompt"],
            function_arguments=sample["inferred_args"]["function_arguments"],
            function_response=sample["inferred_args"]["function_response"],
            assistant_response=sample["inferred_args"]["assistant_response"],
        )
        formatted_samples.append(formatted_sample)

    dataset = datasets.Dataset.from_dict({"input_ids": formatted_samples})
    dataset.save_to_disk(output_dir)


def make_unique(screenshot_dir: Path):
    target_screenshot_dir = screenshot_dir.parent / (screenshot_dir.name + "_unique")
    target_screenshot_dir.mkdir(parents=True, exist_ok=True)

    imgs, filepaths = utils.load_screenshots_from_folder(
        screenshot_dir, downsample=True
    )
    _, unique_filepaths = utils.filter_unique_screenshots(imgs, filepaths)
    utils.copy_imgs_to_target_dir(unique_filepaths, target_screenshot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_vision",
        action="store_true",
        help="Use vision model to extract api signature",
    )
    parser.add_argument(
        "--screenshot_dir", type=str, default="data/screenshots", help="Screenshot dir"
    )
    parser.add_argument(
        "--skip_make_unique",
        action="store_true",
        help="Make unique screenshot dataset dataset before extracting data",
    )
    parser.add_argument("--output_dataset_dir", default="data/dataset", type=str)
    parser.add_argument("--skip_arrow_dataset", action="store_true")
    parser.add_argument("--samples_filepath", default="data/samples.jsonl", type=str)
    args = parser.parse_args()

    if not args.skip_make_unique:
        print("Making unique screenshot dataset")
        screenshot_dir = make_unique(Path(args.screenshot_dir))

    print("Extracting data")
    samples = main(
        use_vision=args.use_vision,
        screenshot_dir=Path(args.screenshot_dir),
        samples_filepath=Path(args.samples_filepath),
    )

    if not args.skip_arrow_dataset:
        print("Compiling arrow dataset")
        compile_arrow_dataset(samples, Path(args.output_dataset_dir))
