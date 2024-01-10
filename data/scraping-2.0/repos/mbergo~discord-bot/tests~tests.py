import sys
import requests
import discord
import openai
import unittest

class TestGPT2DiscordBot(unittest.TestCase):
    discord_token = "sys.argv[1]"
    api_key = "sys.argv[2]"
    auth_key = "sys.argv[3]"
    from_number = "sys.argv[4]"
    
    def test_talk_command(self):
        # set up a mock Discord channel
        ctx = MockDiscordContext()

        # test the talk command
        talk_command(ctx, "Hello")
        response = ctx.sent_messages[0]
        self.assertTrue("Hello" in response)

    def test_predy_criar_canal_command(self):
        room_id = "mock_room_id"
        channel_name = "mock_channel_name"
        headers = {
            "Authorization": f"Bot {self.discord_token}",
            "User-Agent": "MyBot/0.0.1",
            "Content-Type": "application/json",
        }

        # test the predy_criar_canal command
        predy_criar_canal_command(room_id, channel_name, self.discord_token)
        response = requests.post.call_args[1]["headers"]
        self.assertEqual(headers, response)

    def test_predy_whatsapp_command(self):
        # set up a mock Discord channel
        ctx = MockDiscordContext()

        to = "mock_to"
        message = "mock_message"
        headers = {
            "Authorization": f"Bearer {self.auth_key}",
            "Content-Type": "application/json"
        }
        data = {
            "from": self.from_number,
            "to": to,
            "message": message
        }

        # test the predy_whatsapp command
        predy_whatsapp_command(ctx, to, message)
        response = requests.post.call_args[1]["headers"]
        self.assertEqual(headers, response)

class MockDiscordContext:
    def __init__(self):
        self.sent_messages = []

    def send(self, message):
        self.sent_messages.append(message)

if __name__ == '__main__':
    unittest.main()
