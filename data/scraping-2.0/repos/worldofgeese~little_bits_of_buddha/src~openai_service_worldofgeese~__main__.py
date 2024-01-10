import json
import logging
import time
import warnings

import requests
import trio
from dapr.clients import DaprClient
from dapr.ext.fastapi import DaprApp
from fastapi import FastAPI
from hypercorn.config import Config
from hypercorn.trio import serve
from pydantic import BaseModel
from simpleaichat import AIChat
from trio import TrioDeprecationWarning, to_thread

from openai_service_worldofgeese import init_secrets

# Filter out any deprecation warnings
warnings.filterwarnings(action="ignore", category=TrioDeprecationWarning)

app = FastAPI()
dapr_app = DaprApp(app)
config = Config()
config.bind = ["0.0.0.0:8080"]


class CloudEvent(BaseModel):
    datacontenttype: str
    source: str
    topic: str
    pubsubname: str
    data: dict
    id: str
    specversion: str
    tracestate: str
    type: str
    traceid: str


@dapr_app.subscribe(pubsub="scaleway-redis-cluster-pubsub", topic="messages")
async def messages_subscriber(event: CloudEvent):
    logging.info(f"Received message: {event.data}")
    ai = AIChat(
        model="gpt-4",
        system="You are the Buddha. You teach only the Dhamma, only what is fundamental to the holy life as you profess in the Simsapa Sutta. You speak in the style of the Tathagata, the Buddha, the Awakened One of the Early Buddhist Canon.",
        console=False,
    )
    text = event.data.get("text")
    response = {
        "chat_id": event.data.get("chat_id"),
        "text": ai(text),
    }

    # Publish the message to the message bus
    with DaprClient() as dapr_client:
        dapr_client.publish_event(
            pubsub_name="scaleway-redis-cluster-pubsub",
            topic_name="responses",
            data=json.dumps(response),
            data_content_type="application/json",
        )

    return {"success": True}


def wait_for_dapr_ready(
    dapr_port=3500, retries=20, delay=2, task_status=trio.TASK_STATUS_IGNORED
):
    """Wait for the Dapr sidecar to be ready.

    Arguments:
    dapr_port -- The port on which the Dapr sidecar is listening.
    retries -- The number of times to check if Dapr is ready before giving up.
    delay -- The delay between checks.
    """
    dapr_url = f"http://localhost:{dapr_port}/v1.0/healthz"
    for _ in range(retries):
        try:
            response = requests.get(dapr_url)
            if response.status_code == 204:
                print("Dapr is ready.")
                task_status.started()
                return
        except Exception as e:
            print(f"Dapr is not ready yet: {e}")
        time.sleep(delay)

    raise RuntimeError("Dapr sidecar is not ready.")


# Define an async wrapper for wait_for_dapr_ready that reports when it's done
async def async_wait_for_dapr_ready(task_status=trio.TASK_STATUS_IGNORED):
    result = await to_thread.run_sync(wait_for_dapr_ready)
    task_status.started()  # signal that this task is ready
    return result


# Define an async wrapper for init_secrets that reports when it's done
async def async_init_secrets(task_status=trio.TASK_STATUS_IGNORED):
    result = await to_thread.run_sync(init_secrets)
    task_status.started()  # signal that this task is ready
    return result


async def main():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(serve, app, config)
        await nursery.start(async_wait_for_dapr_ready)
        await nursery.start(async_init_secrets)


if __name__ == "__main__":
    trio.run(main)
