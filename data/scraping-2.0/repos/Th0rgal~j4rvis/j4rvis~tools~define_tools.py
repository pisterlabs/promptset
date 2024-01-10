import getpass
from typing import Any
from langchain.agents import Tool
from langchain.utilities import (
    WikipediaAPIWrapper,
    GoogleSearchAPIWrapper,
    PythonREPL,
)
from .parsers import remove_code_block
from .email_tools import send_email_builder
from .calendar_tools import calendar_tool
from .misc_tools import shell_tool_runner, _get_platform
from .unsplash_tools import search_images_runner_builder
from .pdf_tools import html_to_pdf_runner
from .document_tools import document_tool_builder
from langchain.chat_models import ChatOpenAI


def define_tools(config: dict[str, Any]):
    return [
        Tool(
            name="Email Sender",
            func=send_email_builder(
                config["email"]["username"],
                config["email"]["password"],
                config["email"]["server_url"],
                config["email"]["port"],
            ),
            description=(
                "A way to send emails from your own account, j4rvis.assistant@gmail.com."
                "Input should be a json object with string fields 'to_email', 'subject', 'body' and an array of strings field 'files'."
                "The body contains your message in HTML format. It must be well-formulated and classy."
                "The files is a list of paths of files from your computer you want to send at attachments."
                "You must specify in it that you are Mr. Thomas Marchand's assistant. "
                "The output will be a confirmation the email was sent or an error."
            ),
        ),
        Tool(
            name="Calendar",
            func=lambda txt: calendar_tool(config, txt),
            description=(
                "A Calendar Tool to create events and retrieve events within a specific date range on your employer calendar. "
                "The input should be a JSON object with 'action' key and optional 'data' key. "
                'To create an event: \'{"action": "create_event", "data": {"summary": "My Event", "dtstart": "2023-06-01T12:00:00", "dtend": "2023-06-01T13:00:00"}}\'. '
                'To get events: \'{"action": "get_events", "data": {"from_dt": "2023-06-01", "to_dt": "2023-06-30"}}\''
            ),
        ),
        Tool(
            name="Wikipedia",
            func=WikipediaAPIWrapper().run,
            description=(
                "A wrapper around Wikipedia. "
                "Useful for when you need to answer general questions about "
                "people, places, companies, facts, historical events, or other "
                "subjects. Input should be a search query."
            ),
        ),
        Tool(
            name="Google Search",
            func=lambda txt: str(GoogleSearchAPIWrapper().results(txt, 10)),
            description=(
                "A wrapper around the Google search engine. Useful for when "
                "you need to answer questions about current events. "
                "Input should be optimized for a search engine."
                "It returns the 10 first results as a json array of "
                "objects with the following keys: "
                "snippet - The description of the result."
                "title - The title of the result."
                "link - The link to the result."
            ),
        ),
        Tool(
            name="Unsplash Search",
            func=search_images_runner_builder(config),
            description=(
                "A wrapper around an images search engine. Useful for when "
                "you need to find beautiful illustrations for a simple input. "
                "Output is an array of json objects with description, full_image and small_image urls. "
                "Use the small image url when answering in markdown."
            ),
        ),
        Tool(
            name="Python REPL",
            func=lambda txt: PythonREPL().run(remove_code_block(txt)),
            description=(
                "A Python shell. Use this to execute python commands. "
                "Input should be a valid python command. "
                "If you want to see the output of a value, you should print it out "
                "with `print(...)`."
            ),
        ),
        Tool(
            name="Terminal",
            func=shell_tool_runner,
            description=(
                f"Run shell commands on this {_get_platform()} machine and returns the output."
                f"Use this as your own machine, you are connected as '{getpass.getuser()}'."
                "Useful when you need to manage and write files or when you need to query the internet."
                "Input must be a json object with a list of commands, for example:"
                '{"commands": ["echo \'Hello World!'
                '", "time"]}'
            ),
        ),
        Tool(
            name="Document Generator",
            func=document_tool_builder(
                ChatOpenAI(model_name="gpt-4", temperature=0.15)
            ),
            description=(
                "An AI document generator that generates an HTML and CSS document "
                "based on the provided document description. The documents adhere to a clean "
                "and professional design aesthetic, fitting within an A4 page. "
                "The document is outputted as two separate files: index.html and styles.css. "
                "The input should be a precise document description string that includes all "
                "necessary information, as this tool does not have access to any user data "
                "not provided in the input. This means any personal or banking information "
                "needed in the document must be specified in the input description. "
                "The output will be a string message indicating the success of the operation "
                "and paths to the generated HTML and CSS files."
            ),
        ),
        Tool(
            name="HTML to PDF",
            func=html_to_pdf_runner,
            description=(
                "A tool to convert HTML and CSS files to a PDF file. "
                "Input is a JSON object with two keys 'html' and 'css', both strings indicating the paths to the files. "
                "'output' key is optional and specifies the output PDF file path. If not provided, output.pdf will be used. "
                "Output will be a string message indicating the success or failure of the operation."
            ),
        ),
    ]
