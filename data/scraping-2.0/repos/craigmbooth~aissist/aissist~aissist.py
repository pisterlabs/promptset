import os
import sys

import openai
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

from .config import Config
from .exceptions import AIssistError
from .model import Model, OpenAIMessage
from .spinner import Spinner
from .version import __version__

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    print("Please put your API key in the OPENAI_API_KEY environment variable.")
    sys.exit(1)


def prompt_continuation(
    width: int, line_number: int, is_soft_wrap: int  # pylint: disable=W0613
) -> str:
    """When the user is typing a multi-line prompt, this function is called to
    determine what the prompt should look like on the next line.

    n.b. there are unused parameters to this function, but they are required by
    the PromptSession.prompt_continuation interface."""
    return "." * (width - 1) + " "


def print_streaming_message(
    model: Model, messages: list[OpenAIMessage], config: Config
) -> OpenAIMessage:
    """Prints a message that is being streamed from the API"""

    new_message_str = ""
    for printable_chunk in model.stream_call(messages, config):
        sys.stdout.write(printable_chunk)
        sys.stdout.flush()
        new_message_str += printable_chunk
    print("\n")
    new_message: OpenAIMessage = {"role": "assistant", "content": new_message_str}
    return new_message


def print_message(
    model: Model, messages: list[OpenAIMessage], config: Config
) -> OpenAIMessage:
    """Prints a message that is being returned from the API"""

    spinner = Spinner()
    spinner.start()

    new_message = model.call(messages, config)
    spinner.stop()
    print(" ")
    config.code_formatter.highlight_codeblocks(new_message["content"])
    print("\n")

    return new_message


def loop(config: Config, model: Model) -> None:
    """Main loop for the program"""

    # Move the fdollowing two lines into config, too
    style = Style.from_dict({"prompt": "#aaaaaa"})
    session: PromptSession = PromptSession(style=style)

    messages: list[OpenAIMessage] = [{"role": "system", "content": config.prompt}]

    get_completion_function = (
        print_message if config.get("no-stream") is True else print_streaming_message
    )

    while True:
        result = session.prompt(
            ">>> ", multiline=True, prompt_continuation=prompt_continuation
        )
        print()

        messages.append({"role": "user", "content": result})

        new_message = get_completion_function(model, messages, config)

        messages.append(new_message)


def main() -> None:
    print(f"AIssist v.{__version__}. ESCAPE followed by ENTER to send. Ctrl-D to quit")
    print("\n")

    config = Config()
    model = Model(config.get("model"))

    while True:
        try:
            loop(config, model)
        except KeyboardInterrupt:
            # Ctrl-C
            print("Ctrl-C stops current activity. Ctrl-D exits.")
        except EOFError:
            # Ctrl-D
            sys.exit(0)
        except AIssistError as e:
            print(e)
            sys.exit(1)
        except openai.error.OpenAIError as e:
            print(f"Received Error from openai: {str(e)}")
            sys.exit(1)
        except Exception:  # pylint: disable=W0706
            # If we didn't catch the error with one of the specific exceptions above,
            # print the stack trace and exit with a non-zero exit code.
            raise


if __name__ == "__main__":
    main()
