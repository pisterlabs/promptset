import os
import sys

from pathlib import Path
from typing import Literal

import openai

from discord_to_telegram_forwarder.config.config import Config
from discord_to_telegram_forwarder.deps.deps import Deps
from loguru import logger
from telethon import TelegramClient

from lessmore.utils.configure_loguru.format_as_json_colored.format_as_json_colored import format_as_json_colored
from lessmore.utils.load_pydantic_settings.load_pydantic_settings import load_pydantic_settings


def init_deps(env: Literal["test", "prod"] = "test", log_level="DEBUG") -> Deps:
    # - Init config

    config = load_pydantic_settings(
        pydantic_class=Config,
        config_source=[
            {
                "type": "file",
                "is_required": False,
                "value": "{root}/config.secrets.{env}.yaml",
            },
            # "environment_variables",
        ],
        context={
            "root": str(Path(__file__).parent / "../config"),
            "env": os.environ.get("DISCORD_TO_TELEGRAM_FORWARDER_ENV", env),
        },
    )

    # - Configure openai

    openai.api_key = config.openai_api_key

    # - Init logger

    logger.remove()
    logger.add(sink=sys.stdout, level=log_level, format=format_as_json_colored)

    # - Return context

    os.makedirs(str(Path(__file__).parent / "../../data/dynamic"), exist_ok=True)

    return Deps(
        config=config,
        cache={},
        telegram_bot_client=TelegramClient(
            session=str(Path(__file__).parent / "../../data/dynamic/telegram_bot.session"),
            api_id=int(config.telegram_api_id),
            api_hash=config.telegram_api_hash,
        ),
        telegram_user_client=TelegramClient(
            session=str(Path(__file__).parent / "../../data/dynamic/telegram_user.session"),
            api_id=int(config.telegram_api_id),
            api_hash=config.telegram_api_hash,
        ),
    )
