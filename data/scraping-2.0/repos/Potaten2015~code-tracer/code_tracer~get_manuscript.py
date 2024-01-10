from dotenv import load_dotenv
import openai
import json
import os
import tiktoken

load_dotenv()


def create_payload(change_files, config, logger):
    logger.info("Creating GPT payload...")
    changes = {
        filename: {"beginning": values[0]["content"], "end": values[-1]["content"]}
        for filename, values in change_files.items()
    }

    payload = {
        "video_catch_phrase": config.get("video_catch_phrase"),
        "total_progress": config.get("total_progress"),
        "project_name": config.get("project_name"),
        "sessions": config.get("render_sessions"),
        "changes": changes,
        "seconds": config.get("video_length"),
        "sub_project": config.get("sub_project"),
    }

    return payload


def get_manuscript(payload, config, logger):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    logger.info("Using GPT to generate manuscript...")
    context = open(os.path.expanduser(config.get("context_filepath")), "r").read()
    tokens = {}
    for filename, file_changes in payload["changes"].items():
        tokens[filename] = len(encoding.encode(json.dumps(file_changes)))
    encoded = encoding.encode(f"{json.dumps(payload)} {context}")
    logger.info(f"Total tokens: {len(encoded)}")

    reducing = True
    while reducing:
        encoded = encoding.encode(f"{json.dumps(payload)} {context}")
        logger.info(f"Total tokens: {len(encoded)}")
        print("Would you like to remove any files from the GPT payload?")
        for i, (filename, token_count) in enumerate(tokens.items()):
            print(f"{i + 1}. {filename} ({token_count} tokens)")
        print("0. No")
        selection = input("Enter a number: ")
        try:
            selection = int(selection)
            if selection == 0:
                reducing = False
            else:
                filename = list(tokens.keys())[selection - 1]
                proceed = input(f"Are you sure you want to remove {filename}? (y/n) ")
                if proceed == "y":
                    del tokens[filename]
                    del payload["changes"][filename]
        except (ValueError, IndexError):
            print("Invalid selection.")

    encoded = encoding.encode(f"{json.dumps(payload)} {context}")
    logger.info(f"Total tokens: {len(encoded)}")

    if len(encoded) > 3500:
        model = "gpt-3.5-turbo-16k"
    else:
        model = "gpt-3.5-turbo"
    openai.api_key = config.get("openai_api_key")
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )

    output_location = config.get("video_output_dir")
    gpt_api_response_filepath = os.path.join(output_location, 'gpt_api_response.json')
    with open(gpt_api_response_filepath, "w") as out:
        json.dump(completion, out, indent=4)
        logger.info(f'GPT api response written to file: {gpt_api_response_filepath}')
    content = completion.choices[0].message.content
    gpt_content_filepath = os.path.join(output_location, 'gpt_content.json')
    response_json = json.loads(content)
    with open(gpt_content_filepath, "w") as out:
        json.dump(response_json, out, indent=4)
        logger.info(f'GPT message content written to file: {gpt_content_filepath}')
    config.set("total_progress", response_json["total_progress"])
    config.write()

    manuscript = response_json["manuscript"]
    manuscript = manuscript.replace("`", "'")
    return manuscript
