import json
import asyncio

from django.conf import settings
from openai import OpenAI

from channels.generic.websocket import AsyncWebsocketConsumer

client = OpenAI(api_key=settings.OPENAI_API_KEY)

# An initial message to send to OpenAI that explains the concept of the
# game and the kind of responses we want.
system_prompt = """
There is a game in which two players both work to complete a simple drawing
according to a series of prompts that build on one another.

Each player receives a series of distinct prompts.  The prompts should be
related in that they both apply to the same drawing, but the prompts
should be designed so that they apply to different parts of the drawing.

For instance, one player might receive a series of prompts around drawing
the sky, while the other player is drawing the ground.  This is only an example.
Please generate your own theme.

Please generate two series of prompts in JSON format, one for each player.

The JSON should have a player1 key and a player2 key.  The value of each key
should be an array of strings.
"""


class DrawConsumer(AsyncWebsocketConsumer):
    # The game prompts are fetched from OpenAI when the first player connects.
    # We need to avoid re-fetching when the second player connects, and also
    # make sure both players are sent their prompts at the same time.
    # The first player to connect will acquire the fetch_lock while the second
    # player will block on fetch_lock until the prompts are ready.
    fetch_lock = asyncio.Lock()

    prompts_fetched = False

    countdown_time = 30
    countdown_active = False

    # Track the connection count as an easy way to know which set of prompts
    # to send when a new connection is established.
    connection_count = 0

    player1_prompts = []
    player2_prompts = []

    async def start_countdown(self):
        """
        Starts a countdown timer and sends an update of how many seconds
        remain to each connected Websocket client.

        The timer will restart when it reaches zero.
        """
        if not DrawConsumer.countdown_active:
            DrawConsumer.countdown_active = True
            while DrawConsumer.countdown_time > 0:
                await asyncio.sleep(1)
                await self.channel_layer.group_send(
                    "draw_group",
                    {
                        "type": "countdown_message",
                        "countdown": DrawConsumer.countdown_time,
                    },
                )
                DrawConsumer.countdown_time -= 1

                if DrawConsumer.countdown_time == 0:
                    DrawConsumer.countdown_time = 30

    async def fetch_prompts(self):
        """
        Fetch prompts from the OpenAI API.

        The first caller will trigger the API request while subsequent calls
        block on the fetch_lock, eventually returning immediately because the
        prompts_fetched bool has been set to True.
        """
        async with DrawConsumer.fetch_lock:
            if not DrawConsumer.prompts_fetched:
                DrawConsumer.prompts_fetched = True

                completion = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "user",
                            "content": system_prompt,
                        }
                    ],
                )

                data = json.loads(completion.choices[0].message.content)

                DrawConsumer.player1_prompts = data["player1"]
                DrawConsumer.player2_prompts = data["player2"]

    async def connect(self):
        """
        Set up Websocket connection and make OpenAI request if this is the
        first connection.  Distributes prompts to players.
        """
        DrawConsumer.connection_count += 1

        await self.channel_layer.group_add("draw_group", self.channel_name)
        await self.fetch_prompts()

        await self.accept()

        prompts = [DrawConsumer.player1_prompts, DrawConsumer.player2_prompts]

        await self.send(
            text_data=json.dumps(
                {
                    "type": "prompts",
                    "prompts": prompts[DrawConsumer.connection_count % 2],
                }
            )
        )

        asyncio.create_task(self.start_countdown())

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("draw_group", self.channel_name)

    async def receive(self, text_data):
        """
        Broadcast messages received from one player back to themselves and to
        the other player in the game.
        """
        data = json.loads(text_data)
        await self.channel_layer.group_send(
            "draw_group", {"type": "draw_message", "message": data}
        )

    async def draw_message(self, event):
        message = event["message"]
        await self.send(text_data=json.dumps(message))

    async def countdown_message(self, event):
        countdown = event["countdown"]
        message = {"type": "countdown", "countdown": countdown}
        await self.send(text_data=json.dumps(message))
