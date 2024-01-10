import base64
import io
import json

import structlog
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from openai.types.chat.completion_create_params import ResponseFormat
from PIL import Image

from browser import Browser

client = AsyncOpenAI()

CHEAP = "gpt-3.5-turbo-1106"
GOOD = "gpt-4-1106-preview"
VISION = "gpt-4-vision-preview"

log = structlog.get_logger()


async def convert_png_to_jpeg_base64(path: str) -> str:
    log.debug("Converting PNG to JPEG")
    # Re-load the PNG image from the filesystem
    with open("example.png", "rb") as image_file:
        image = Image.open(image_file)
        # If the image has an alpha channel, convert it to RGB before saving as JPEG
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            # Convert the image to RGB, discarding the alpha channel
            image = image.convert("RGB")
        # Convert the image to JPEG
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        # Encode the JPEG image to base64
        img_str = base64.b64encode(buffered.getvalue())
        log.debug("Image converted")
        return img_str.decode("utf-8")


async def predict_marketing_email(content: str) -> dict:
    soup = BeautifulSoup(content, "html.parser")
    # links = {a['href']: a.get_text() for a in soup.find_all('a')}
    links = [a.get_text() for a in soup.find_all("a")]
    plain_text = soup.get_text()[:4000]
    log.debug("links", links=links)
    links_str = str(links)
    response = await client.chat.completions.create(
        timeout=10,
        model=CHEAP,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that determines the most likely link for"
                    " unsubscribing. The user will paste in a list of links from a"
                    " page, only the inner text but not the URL, as well as the plain"
                    " text of the email, and you will respond in JSON with the most"
                    " likely text for the unsubscribe link if it appears to be a"
                    " marketing email that will clutter an inbox. If the email is a"
                    " receipt or other important information, set <text> to null. "
                    ' Respond in {"unsubscribe_link": <text>}. Set <text> to `null` if'
                    " there is no link that appears to be for unsubscribing or managing"
                    " email preferences."
                ),
            },
            {"role": "user", "content": f"Links: {links_str}\n\nText: {plain_text}"},
        ],
        response_format=ResponseFormat(type="json_object"),
    )
    log.debug("Marketing email predicted")
    return json.loads(response.choices[0].message.content)

async def clean_up_json(content: str) -> dict:
    log.debug("Cleaning up JSON", content=content)
    response = await client.chat.completions.create(
        timeout=10,
        model=CHEAP,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that reads the provided input from the user"
                    " and returns correctly formatted JSON."
                ),
            },
            {"role": "user", "content": content},
        ],
        response_format=ResponseFormat(type="json_object"),
    )
    log.debug("JSON cleaned")
    return json.loads(response.choices[0].message.content)


async def run_unsubscribe_loop(url: str):
    log.debug("Running unsubscribe loop")
    browser = Browser()
    await browser.start()
    await browser.go_to(url)
    done = False
    previous_actions = []
    while not done:
        log.debug("previous actions", actions=previous_actions)
        await browser.screenshot()
        snapshot = await browser.snapshot()
        snapshot_str = str(snapshot)
        log.debug("snapshot", snapshot=snapshot)
        base64_image_str = "data:image/jpeg;base64," + await convert_png_to_jpeg_base64(
            "example.png"
        )
        actions_str = str(previous_actions)
        response = await client.chat.completions.create(
            timeout=60,
            model=VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I have provided a screenshot of an unsubscribe page,"
                                " as well as the accessibility tree for the page."
                                " Respond in JSON with the next step the user must"
                                ' perform, such as {"role": "link", "name":'
                                ' "unsubscribe", "action": "click", "data" null} or'
                                ' {"role": "textbox", "name": "email", "action":'
                                ' "enter_text", "data": "maccam912@gmail.com"} to put'
                                " that email address in a text input element. If the"
                                " screenshot indicates that the user has now"
                                ' successfully unsubscribed, respond with {"role":'
                                ' null, "action": "done", "locator": null, "data":'
                                " null}. Up to this point the user has performed the"
                                f" following actions: {actions_str}\n\nAccessibility"
                                f" tree: {snapshot_str}"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": base64_image_str}},
                    ],
                }
            ],
            max_tokens=300,
        )
        vision_response = response.choices[0].message.content
        action = await clean_up_json(vision_response)
        log.debug("next action", action=action)
        if "action" in action and action["action"] in ["click", "done"]:
            if action["action"] == "done":
                done = True
            elif action["action"] == "click":
                await browser.click(action["role"], action["name"])
                previous_actions += [action]
        else:
            # What action was suggested here?
            log.debug("unknown action", action=action)
            return
        await browser.screenshot()
        log.debug("Got screenshot")
    return
