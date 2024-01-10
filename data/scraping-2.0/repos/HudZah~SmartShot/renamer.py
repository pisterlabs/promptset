import os

from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64


def find_latest_screenshot(directory):
    """Find the most recently added screenshot in the directory."""
    all_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png")
    ]
    latest_file = max(all_files, key=os.path.getctime)
    print(f"Found latest file: {latest_file}")
    return latest_file


def analyze_image(image_path):
    """Analyze the image using GPT-4 API and return the description."""

    client = OpenAI()

    with Image.open(image_path) as img:
        IMG_RES = 512
        W, H = img.size
        img = img.resize((IMG_RES, int(IMG_RES * H / W)))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a screenshot of something on my screen",
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{img_base64}",
                    },
                    {
                        "type": "text",
                        "text": "Provide me a concise and descriptive file name for this image. Do not add any file extension. Be as descriptive as possible and keep it shorter than 6 words.",
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def rename_image(image_path, description):
    """Rename the image file based on the description."""
    directory, filename = os.path.split(image_path)
    new_filename = "_".join(description.split()) + ".png"
    new_path = os.path.join(directory, new_filename)
    os.rename(image_path, new_path)
    return new_path


def has_been_processed(filename, processed_log):
    """Check if a file has already been processed."""
    if os.path.exists(processed_log):
        with open(processed_log, "r") as log:
            if filename in log.read():
                return True
    return False


def mark_as_processed(filename, processed_log):
    """Mark a file as processed by logging its name."""
    with open(processed_log, "a") as log:
        log.write(filename + "\n")


def main():
    # Modify these paths to your own
    DIRS = [
        "/Users/username/path_to_screenshots",
        "/Users/hudzah/some_folder/screenshot-renamer/processed.log",
    ]

    screenshots_dir = DIRS[0]
    processed_log = DIRS[1]

    latest_screenshot = find_latest_screenshot(screenshots_dir)

    if not has_been_processed(latest_screenshot, processed_log):
        description = analyze_image(latest_screenshot)
        new_path = rename_image(latest_screenshot, description)
        mark_as_processed(new_path, processed_log)
        print(
            f"Renamed '{latest_screenshot}' to '{new_path}' and added to processed log"
        )


if __name__ == "__main__":
    import sys

    print(sys.executable)

    main()
