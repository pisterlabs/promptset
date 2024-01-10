import asyncio
import base64
import re
from datetime import datetime
from typing import AsyncIterable

import openai
from fastapi_poe import PoeBot
from fastapi_poe.types import (
    ProtocolMessage,
    QueryRequest,
    ReportFeedbackRequest,
    SettingsRequest,
    SettingsResponse,
)
from firebase_admin import db, storage
from sse_starlette.sse import ServerSentEvent

MAX_PER_DAY = 3
SETTINGS = SettingsResponse(
    context_clear_window_secs=60 * 60, allow_user_context_clear=True
)

SYS_PROMPT = "You are ImaGen, advanced chat bot capable of generating images and chatting with the user. When the user wants to make an image, use the /gen command to do so. If the user types '/gen', you should infer that they want to generate an image, so use the /gen command. DO NOT share image links, DO NOT recommend other platforms or services to generate images, DO NOT say things like [insert image here...], [generated image...], [image from /gen command], etc. and DO NOT say you can't generate an image, just use the /gen command to do so. You can use it by typing '`/gen <prompt>`', where <prompt> is the prompt to generate an image with, inferred from your conversation with the user - this is the only way you can generate images. Do not use the /gen command until you confirm that the user wants to make an image with the prompt you have inferred/suggested. If the user did not provide a detailed enough prompt, you can suggest 2 more detailed prompts based on their input (Example prompt: 'Color photo of a cat on a car, fluffy fur, curious expression, busy street, tall buildings, Nikon D850, Fujifilm Pro 400H, 50mm lens, panning, Wes Anderson, Christopher Nolan, Annie Leibovitz') - when suggesting multiple prompt options, do not use the /gen, only use it when the user chooses a specific one. Examples: [{'user': 'make an image of a city at sunset, bright colors and details', 'assistant': '`/gen a city at sunset, bright colors and details`'}, {'user': 'a litter of kittens cradled in a basket', 'assistant': '`/gen a litter of kittens cradled in a basket`'}, {'user': '/gen a night sky', 'assistant': '`/gen a beautiful night sky full of stars, with a comet flying by`'}]]"


class ImaGenBot(PoeBot):
    async def get_response(self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        """Return an async iterator of events to send to the user."""
        yield self.meta_event(
            content_type="text/markdown",
            linkify=False,
            refetch_settings=False,
            suggested_replies=False,
        )

        # convert to openai usable convo
        convo = self.convert_query(query.query)

        # Normal conversation
        full_reply = ""
        for chunk in openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=convo, temperature=0.2, stream=True
        ):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                full_reply = full_reply + content
                yield self.text_event(content)

        # How many images has this user generated today
        used = await self.get_free_used(query.user_id)

        if "`/gen" in full_reply:
            # yield self.done_event()
            # Show wall if >MAX_PER_DAY
            if used >= MAX_PER_DAY:
                yield self.text_event("\n\nDrawing the image...")
                yield self.text_event("\n\n***\n\n")
                yield self.text_event(
                    "You've used your 3 free image generations for the day. Come back tomorrow!"
                )
                yield self.done_event()

            else:
                regex = re.search("`(.+?)`", full_reply)
                if regex and len(regex.group(1).replace("/gen", "")) > 6:
                    prompt = regex.group(1).replace("/gen", "")
                    yield self.text_event("\n\nDrawing the image...")
                    yield self.text_event("\n\n***\n\n")

                    try:
                        images = await self.gen_image(
                            prompt, query.message_id, query.user_id
                        )
                        for idx, url in enumerate(images):
                            yield self.text_event(
                                "**Variation " + str(idx + 1) + "** · "
                            )
                            yield self.text_event("[Image link](" + url + ")\n\n")
                            yield self.text_event(
                                "![Variation " + str(idx + 1) + "](" + url + ")\n\n"
                            )
                            if idx < len(images) - 1:
                                yield self.text_event("\n\n***\n\n")

                        # Record generated image count
                        now = datetime.now()
                        current_date = now.strftime("%m-%d-%Y")

                        userRef = db.reference(
                            "users/" + query.user_id + "/" + current_date
                        )
                        userRef.set({"freeUsed": used + 1})

                        # Show user how many are left
                        yield self.text_event("\n\n***\n\n")
                        yield self.text_event(
                            "You have **"
                            + str(MAX_PER_DAY - used - 1)
                            + "** remaining free credits today. "
                        )

                        yield self.done_event()

                    except ImageException as e:
                        yield self.text_event(e.msg)
                        yield self.text_event(" Failed images do not use credits. ")
                        yield self.text_event(
                            "You have **"
                            + str(MAX_PER_DAY - used)
                            + "** remaining free credits today."
                        )
                        yield self.error_event(e.msg)

                else:
                    yield self.done_event()
        else:
            yield self.done_event()

    async def on_feedback(self, feedback: ReportFeedbackRequest) -> None:
        """Called when we receive user feedback such as likes."""
        feedbackRef = db.reference(
            "feedback/" + feedback.user_id + "/" + feedback.conversation_id
        )
        feedbackRef.update({feedback.message_id: feedback.feedback_type})

    async def get_settings(self, settings: SettingsRequest) -> SettingsResponse:
        """Return the settings for this bot."""
        return SETTINGS

    async def get_free_used(self, id: str) -> int:
        now = datetime.now()
        current_date = now.strftime("%m-%d-%Y")

        userRef = db.reference("users/" + id + "/" + current_date)

        userDoc = userRef.get()

        if userDoc is None:
            # Set image tag as zero and init user
            userRef.set({"freeUsed": 0})
            return 0

        used = userDoc["freeUsed"]
        return used

    async def gen_image(self, prompt: str, id: str, user_id: str):
        try:
            img_urls = []

            images = await openai.Image.acreate(
                prompt=prompt, n=2, size="512x512", response_format="b64_json"
            )

            for idx, img in enumerate(images.data):
                # upload image to firebase under user id folder
                file_name = user_id + "/" + str(id + "-" + str(idx) + ".jpg")
                img_data = base64.b64decode(img.b64_json)
                url = await self.upload_to_firebase_async(file_name, img_data)
                print(url)
                img_urls.append(url)

            return img_urls
        except openai.error.OpenAIError as e:
            raise ImageException(e.error.message)

    async def upload_to_firebase_async(self, file_name, img_b64_json) -> str:
        bucket = storage.bucket()
        blob = bucket.blob(file_name)

        await asyncio.to_thread(
            blob.upload_from_string, img_b64_json, content_type="image/jpg"
        )
        blob.make_public()
        return blob.public_url

    def convert_query(self, query: [ProtocolMessage]) -> []:
        messages = []
        messages.append({"role": "user", "content": SYS_PROMPT})
        for message in query[-5:]:
            if message.role == "bot":
                if "***" in message.content:
                    # Strip message gen info
                    content = message.content.split("***")[0].replace(
                        "Drawing the image...", ""
                    )
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "assistant", "content": message.content})
            else:
                messages.append({"role": message.role, "content": message.content})
        return messages


class ImageException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
