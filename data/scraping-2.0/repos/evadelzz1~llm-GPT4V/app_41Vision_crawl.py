from dotenv import load_dotenv
from openai import OpenAI
import subprocess
import base64, json, os

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

client = OpenAI()
client.timeout = 10

def image_b64(image):
    with open(image, "rb") as f:
        return base64.b64encode(f.read()).decode()


def full_screenshot(url, output_path):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)

    driver.get(url)
    time.sleep(1)  # Give the page some time to load

    total_height = driver.execute_script("return document.body.parentNode.scrollHeight")

    driver.execute_script("window.scrollTo(0, document.body.parentNode.scrollHeight);")

    time.sleep(1)
    driver.set_window_size("1920", total_height)

    time.sleep(1)
    driver.save_screenshot(output_path)

    driver.quit()

print("\nEx")
print("\n1) What is the weather like in san francisco?")
print("\n2) What is the current stock price of tesla?")

prompt = input("\n\nYou: ")

messages = [
    {
        "role": "system",
        "content": "You are a web crawler. Your job is to give the user a URL to go to in order to find the answer to the question. Go to a direct URL that will likely have the answer to the user's question. Respond in the following JSON fromat: {\"url\": \"<put url here>\"}",
    },
    {
        "role": "user",
        "content": prompt,
    }
]

while True:
    while True:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            max_tokens=1024,
            response_format={"type": "json_object"},
            seed=2232,
        )

        message = response.choices[0].message
        message_json = json.loads(message.content)
        url = message_json["url"]

        messages.append({
            "role": "assistant",
            "content": message.content,
        })

        print(f"Crawling {url}")

        output_path = "./data/screenshot.png"
        
        if os.path.exists(output_path):
            os.remove(output_path)

        # result = subprocess.run(
        #     ["node", "screenshot.js", url],
        #     capture_output=True,
        #     text=True
        # )

        # exitcode = result.returncode
        # output = result.stdout

        full_screenshot(url, output_path)

        if not os.path.exists(output_path):
            print("ERROR: Trying different URL")
            messages.append({
                "role": "user",
                "content": "I was unable to crawl that site. Please pick a different one."
            })
        else:
            break

    b64_image = image_b64("./data/screenshot.png")

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "Your job is to answer the user's question based on the given screenshot of a website. Answer the user as an assistant, but don't tell that the information is from a screenshot or an image. Pretend it is information that you know. If you can't answer the question, simply respond with the code `ANSWER_NOT_FOUND` and nothing else.",
            }
        ] + messages[1:] + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{b64_image}",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ],
        max_tokens=1024,
    )

    message = response.choices[0].message
    message_text = message.content

    if "ANSWER_NOT_FOUND" in message_text:
        print("ERROR: Answer not found")
        messages.append({
            "role": "user",
            "content": "I was unable to find the answer on that website. Please pick another one"
        })
    else:
        print(f"GPT: {message_text}")
        prompt = input("\nYou: ")
        messages.append({
            "role": "user",
            "content": prompt,
        })
