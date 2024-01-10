from argparse import ArgumentParser

import openai
import uvicorn

from uvicorn_loguru_integration import run_uvicorn_loguru

from .config import config
from .llm_game import LlmGame
from .llm_twitch_bot import LlmTwitchBot


def main():
    parser = ArgumentParser(
        description='Backend for Twitch-Plays-LLM, an interactive collaborative text-based twitch game'
    )
    sp = parser.add_subparsers(dest='action')
    sp.add_parser('run')
    args = parser.parse_args()

    openai.api_key = config.openai_api_key

    if args.action == 'run':
        run_uvicorn_loguru(
            uvicorn.Config(
                'twitch_plays_llm.app:app',
                host='0.0.0.0',
                port=config.backend_port,
                log_level='info',
                reload=False,
                workers=1,  # We need only 1 worker because otherwise multiple chatbots will be running
            )
        )
    else:
        assert False


if __name__ == '__main__':
    main()
