import openai
import os

# Get your at --> https://beta.openai.com/account/api-keys
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY environment variable"

# Use your own API ID and HASH, which you can get from https://my.telegram.org/apps
bot_token = os.getenv("BOT_TOKEN")
bot_name = os.getenv("BOT_NAME")

assert bot_token, "Missing BOT_TOKEN environment variable"
assert bot_name, "Missing BOT_NAME environment variable"

if openai.api_type == "azure": # "OPENAI_API_TYPE"
    engine = "gpt-4"
    DEFAULT_MODEL = None
else:
    DEFAULT_MODEL = "gpt-3.5-turbo"
    engine = None

