# Copyright (c) 2023 Rocket Science AG, Switzerland

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""A websocket query server that uses the data retrieval engine to answer questions."""

import argparse
import asyncio
import logging
import sys
import tracemalloc

from loguru import logger

import rrosti.utils.config
from rrosti.chat.chat_session import OpenAI, UserInputLLM
from rrosti.chat.state_machine import execution
from rrosti.llm_api import openai_api_direct
from rrosti.query import logging as qlog
from rrosti.servers import data_retrieval_engine, websocket_query_server
from rrosti.utils import misc
from rrosti.utils.config import config
from rrosti.utils.misc import ProgramArgsBase


class ProgramArgs(websocket_query_server.ProgramArgsMixin, execution.ProgramArgsMixin, ProgramArgsBase):
    temperature: float

    @classmethod
    def _add_args(cls, parser: argparse.ArgumentParser) -> None:
        super()._add_args(parser)
        parser.add_argument(
            "--temperature",
            type=float,
            default=config.openai_api.completion_temperature,
            help="The temperature of the OpenAI Model",
        )


async def main() -> None:
    rrosti.utils.config.load()
    openai_provider = openai_api_direct.DirectOpenAIApiProvider()

    tracemalloc.start()
    misc.setup_logging()

    args = ProgramArgs.parse_args()

    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO if not args.debug_asyncio else logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("websockets").setLevel(logging.INFO)
    # logging.getLogger("openai").setLevel(logging.INFO)

    qlog.ServerStartedEvent.log(args=sys.argv)

    engine: websocket_query_server.QueryEngineBase
    if args.user_simulate_llm:
        logger.info("Starting query server, using a user simulated LLM")
        engine = data_retrieval_engine.DataQueryEngine(llm=UserInputLLM(), openai_provider=openai_provider)
    else:
        logger.info("Starting query server")
        engine = data_retrieval_engine.DataQueryEngine(
            llm=OpenAI(openai_api_direct.DirectOpenAIApiProvider(), temperature=args.temperature),
            openai_provider=openai_provider,
        )

    asyncio.get_event_loop().set_debug(args.debug_asyncio)
    await engine.aserve_forever(args)


if __name__ == "__main__":
    asyncio.run(main())
