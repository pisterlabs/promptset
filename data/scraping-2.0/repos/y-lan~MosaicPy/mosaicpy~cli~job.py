from typing import List, Optional, Type
from pydantic import BaseModel, Field
from mosaicpy.llm.openai.agent import OpenAIAgent
from datetime import datetime
import mosaicpy as mpy


import logging
import fire
from colorama import init, Fore, Style

from mosaicpy.llm.schema import Event

init()


class ColorfulLogger(logging.StreamHandler):
    def emit(self, record):
        log_level = record.levelno
        if log_level == logging.DEBUG:
            color = Fore.CYAN
        elif log_level == logging.INFO:
            color = Fore.GREEN
        elif log_level == logging.WARNING:
            color = Fore.YELLOW
        elif log_level == logging.ERROR:
            color = Fore.RED
        elif log_level == logging.CRITICAL:
            color = Fore.RED + Style.BRIGHT
        else:
            color = Fore.WHITE
        record.msg = color + str(record.msg) + Style.RESET_ALL
        super(ColorfulLogger, self).emit(record)


def print_ai(msg, prefix="GPT: ", end="\n"):
    print(Fore.MAGENTA + Style.BRIGHT + prefix + msg + Style.RESET_ALL, end=end)


logger = logging.getLogger("mosaicpy.llm")
logger.addHandler(ColorfulLogger())


class JobSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The query to search for, support semantic search, must be in Japanese",
    )
    location: Optional[str] = Field(
        None, description="The location to search in (city name or train station)"
    )

    start_date: Optional[datetime] = Field(
        None, description="The start date for the job, in yyyy-mm-dd format"
    )
    end_date: Optional[datetime] = Field(
        None, description="The end date for the job, in yyyy-mm-dd format"
    )
    start_time: Optional[str] = Field(
        None, description="The start time for the job, in 24-hour format"
    )
    end_time: Optional[str] = Field(None, description="The end time for the job, in 24-hour format")
    weekdays: Optional[List[str]] = Field(None, description="List of weekdays for the job")

    min_wage: Optional[int] = Field(None, description="Minimum wage for the job")
    max_wage: Optional[int] = Field(None, description="Maximum wage for the job")

    tags: Optional[List[str]] = Field(None, description="List of tags associated with the job")
    category: Optional[str] = Field(None, description="Job category")
    use_current_location: bool = Field(False, description="Flag to use user's current location")


class JobSearchTool:
    name = "job_search"
    description = "Search for jobs based on various parameters"
    args_schema: Type[BaseModel] = JobSearchSchema

    @staticmethod
    def _run(
        query: str,
        location: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        weekdays: Optional[List[str]] = None,
        min_wage: Optional[int] = None,
        max_wage: Optional[int] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        use_current_location: bool = False,
    ):
        pass


def main(
    multi_round: bool = True,
    gpt4: bool = False,
    stream: bool = True,
    verbose: bool = False,
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    config = mpy.dict(
        sys="""You are a helpful assistant.
Your task is to parse the user's query and convert into a search query for the job search tool.

Today's date is __DATE__""",
        keep_conversation_state=multi_round,
        stream=stream,
        tools=[JobSearchTool()],
        execute_tools=False,
    )

    if gpt4:
        config["model_name"] = "gpt-4-1106-preview"

    bot = OpenAIAgent(**config)

    if stream:
        bot.subscribe(
            Event.NEW_CHAT_TOKEN, lambda data: print_ai(data["content"], prefix="", end="")
        )
        bot.subscribe(Event.FINISH_CHAT, lambda data: print_ai("", prefix="", end="\n"))
    else:
        bot.subscribe(Event.FINISH_CHAT, lambda data: print_ai(data["response"], prefix=""))

    while True:
        try:
            user_input = input("You (or 'exit' to quit): ")
            # Check if the user wants to exit or input is empty
            if user_input.lower() == "exit":
                print("Exiting the chat...")
                break
            elif user_input.strip() == "":
                continue
        except KeyboardInterrupt:
            print("\nExiting the chat...")
            break

        try:
            print_ai("")
            # Get the response from the bot
            bot.chat(user_input)
        except Exception as e:
            print("An error occurred:", e)
            if verbose:
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    fire.Fire(main)
