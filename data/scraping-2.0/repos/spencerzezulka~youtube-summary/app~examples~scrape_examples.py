import json
import subprocess
import tempfile
from langchain.document_loaders import YoutubeLoader
import os

# Run script with link to well-summarized video to generate few-shot example


def main():
    # 1. Ensure standardized filepath
    json_file_path = "examples.json"

    # 2. Handle case of JSON not (yet) in existence
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            examples = json.load(file)
    else:
        examples = []

    # 3. Prompt user for link
    video_url = input("Enter valid YouTube URL: ")

    # 4. Terminate if link already in dataset
    if any(example["link"] == video_url for example in examples):
        raise ValueError("Link already exists in examples.json. Terminating.")

    # 5. Prompt user for well-formatted summary
    def get_multiline_input(editor="nano"):
        # Instruct user
        print(
            "You will be prompted to enter your well-formatted summary in the nano text editor. After you're done, save and exit to continue."
        )

        # Wait for user to read instructions
        input("Press <Enter> to continue...")

        # Use a temporary file to store user input
        with tempfile.NamedTemporaryFile(mode="r+", delete=False) as tmpfile:
            tmpfile_path = tmpfile.name

            # Open the editor
            subprocess.run([editor, tmpfile_path])

            # Read the content of the file
            with open(tmpfile_path, "r") as tmpfile:
                user_input = tmpfile.read()

            # Optionally delete the temporary file if you want to clean up
            os.remove(tmpfile_path)

            return user_input

    user_summary = get_multiline_input()

    # 6. Load metadata
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    youtube_metadata = loader.load()[0].dict()

    # 5. Add example, given that link is not already in data
    new_entry = {
        "link": video_url,
        "fewshotmapping": {"metadata": youtube_metadata, "summary": user_summary},
    }
    examples.append(new_entry)
    with open(json_file_path, "w") as file:
        json.dump(examples, file, indent=4)
    print("Entry to examples.json successful!")


if __name__ == "__main__":
    main()
