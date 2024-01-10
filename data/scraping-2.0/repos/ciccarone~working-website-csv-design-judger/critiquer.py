import subprocess
import os
from openai import OpenAI
import base64
import time
import errno
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

client = OpenAI()

def encode_image(image_path):
    while True:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                raise
            time.sleep(0.1)

def generate_new_line(base64_image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give professional feedback to the designer"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def analyze_image(base64_image, script):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": """
                You are a world-renowned website critiquer and are giving professional feedback to the designer.  State only facts, no fluff.  One sentence is enough.
                """,
            },
        ]
        + script
        + generate_new_line(base64_image),
        max_tokens=100,
    )
    response_text = response.choices[0].message.content
    return response_text

def main():
    script = []

    # Read the CSV file
    df = pd.read_csv(os.path.join(os.getcwd(), "./frames/file.csv"))

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Create a new instance of the Firefox driver
        driver = webdriver.Firefox()

        url = row['url']  # replace 'url' with your column name

        try:
            driver.get(url)

            subprocess.run(['node', 'screenshot.js', url])

            # driver.save_screenshot(f'frames/site.png')

            # path to your image
            image_path = os.path.join(os.getcwd(), './frames/site.png')
            # print(os.path.join(os.getcwd(), "./frames/frame.jpg"))

            # getting the base64 encoding
            base64_image = encode_image(image_path)

            # analyze posture
            print("👀 Analyzing Website...")
            analysis = analyze_image(base64_image, script=script)

            print("👀 My design feedback on: " + url)
            print(analysis)
            df.at[index, 'critique'] = analysis


            # Write DataFrame back to CSV file
            df.to_csv(os.path.join(os.getcwd(), "./frames/file.csv"), index=False)

            # wait for 5 seconds
            time.sleep(1)

            # Close the browser
            driver.quit()

        except Exception:
            print(f"An error occurred with URL: {url}. Skipping...")
            continue
    print("All websites have been analyzed.")

if __name__ == "__main__":
    main()
