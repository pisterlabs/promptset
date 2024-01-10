""" Gpt_client - A simple Python-based OpenAI GPT client.

Compared to the ChatGPT website this tool is more configurable, is only several
hundred lines of open-source code so is easy to understand/follow/modify,
provides standard Linux readline-based command line functionality, and
allows GPT-4 and plug-in options like inserting weblinks without paying the
hefty monthly fee for ChatGPT-Plus.  (However, note it does involve paying fees
for API calls - but for personal use rather than public internet app use this
is literally pennies per month.)

Can be run either as a terminal-based ChatGPT CLI or as a browser-based web-app.

Usage:
    python3 gpt_client.py --help
which lists the two entrypoint commands:
    python3 gpt_client.py cli --help
    python3 gpt_client.py webapp --help
whose command line options are specified in their help listings.

Note you must have your OPENAI_API_KEY environment variable set.

And you must be in an environment with the following Python packages, which can
be installed with the standard pip requirements file.
  * openai (for the core OpenAI calls)
  * beautifulsoup4 (for reading contents of urls)
  * rich (for the markdown/syntax-highlighting formatting in CLI)
  * darkdetect (for putting code syntax or webpage into light/dark mode in CLI)
  * gradio (for the quickie webapp interface)

  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

"""

from cmd import Cmd
import os
import re
import readline
import sys
import urllib

from bs4 import BeautifulSoup
import click
import darkdetect
import gradio as gr
import openai
from rich.console import Console
from rich.markdown import Markdown


openai.api_key = os.environ["OPENAI_API_KEY"]
webapp_state = {}

# defaults for GPT parameters:
MODEL = "gpt-4-1106-preview"
TEMPERATURE = 0.2
TOP_P = 0.1
MAXCHAR = 20000  # recommend 20000 for 8192 tokens in gpt-4.
                 # gpt-4-1106-preview takes 128k tokens (15x above) but let's
                 # still leave at 20000 for now just to protect against mistake.
ALLOW_INJECTIONS = True
DEBUG = False
SYSMESSAGE = ("The following is a conversation with"
    " an AI assistant. The assistant is helpful, creative, friendly. "
    "Its answers are polite but brief, only rarely exceeding "
    "a single paragraph when really necessary to explain a point. "
    "The assistant labels all markdown code snippets with the code "
    "language. Mathematical answers and expressions written by the "
    "assistant are always formatted in unicode characters rather than "
    "latex, using full mathematical notation rather than programming "
    "notation. The assistant only very occasionally uses emojis to "
    "show enthusiasm.")
# defaults for CLI parameters:
PROMPT = "\n\001\033[01;32m\002😃\001\033[37m\033[01;32m\002 Me:\001\033[00m\002 "
GPTPROMPT = "\001\033[01;32m\002🤖\001\033[37m\033[01;36m\002 GPT:\001\033[00m\002 "
CODE_THEME = "monokai" if darkdetect.isDark() else "default"
INTRO = "Params: <params>\n<instructions>\n"
HISTORY_FILE = os.path.expanduser('~/.gpt_history')


def submit_to_gpt(messages, model, temperature, top_p):
    """Send formatted input contents/parameters to OpenAI API completion call.

    Parameters:
        messages         list     chat history of user & agent messages
                                  in OpenAI's list-of-dicts format like
                                  [{"role":___, "content":___}, {...}, ...]
        model            string   OpenAI's "gpt-4", "gpt-3.5-turbo", etc.
        temperature      float    consistency/creativity parameter 0.0-1.0
        top_p            float    sampling parameter 0.0-1.0
    Returns:
        reply            string   latest chatbot reply message
        metadata         dict     dict of non-msg reply info like token counts
    """

    metadata = {}
    try:
        chat = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        metadata["prompt_tokens"] = chat.usage.prompt_tokens
        metadata["completion_tokens"] = chat.usage.completion_tokens
    except openai.OpenAIError as e:
        errormsg = e.response["error"]["message"]
        print(f"OpenAI API Error: {errormsg}")
        return ""
    except openai.error.APIConnectionError as e:
        print(f"Sorry, wasn't able to connect to API for some reason: {e.reason}.")
        return ""
    except openai.error.ServiceUnavailableError as e:
        print(f"Sorry, API appears to be temporarily unavailable: {e.reason}.")
        return ""
    except openai.error.RateLimitError as e:
        print(f"Sorry, hit a too-many-users limit: {e.reason}.")
        return ""
    except openai.error.InvalidRequestError as e:
        print(f"Sorry, hit a too-long-submission limit: {e.reason}.")
        return ""
    except openai.error.Timeout as e:
        print(f"Sorry, request apparently timed out: {e.reason}.")
        return ""
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return ""

    return reply, metadata


def get_url_contents(url, maxchar):
    """Scrape webpage text-only contents from given input URL.

    Parameters:
        url          string    web address to scrape
        maxchar      integer   number of chars at which to truncate returned
                               webapge contents
    Returns:
        webpagetext  string    webpage text-only contents
        skip_input   boolean   success flag to prevent OpenAI call if problem
    """

    skip_input = False

    try:
        req = urllib.request.Request(url)
        html = urllib.request.urlopen(req).read()
        webpagetext = BeautifulSoup(html, "html.parser").get_text()
        webpagetext = webpagetext.replace("\n", " ")
        orig_length_webpagetext = len(webpagetext)
        if orig_length_webpagetext > maxchar:
            user_input = input(
                "Warning: length of webpagetext string "
                f"= {len(webpagetext)} which will rapidly use up your "
                f"tokens; also must be truncated to first {maxchar} "
                "chars.  Continue?  (yN): ")
            if user_input.lower() != "y" and user_input.lower() != "yes":
                skip_input = True
            else:
                webpagetext = (
                    "GPT please note that due to length, "
                    f"webpage truncated to first {maxchar} characters,"
                    f"about {maxchar/orig_length_webpagetext*100.0}%, "
                    "which may affect your interpretation of it:\n"
                    "------------------\n") + webpagetext
                webpagetext = webpagetext[:maxchar]
    except urllib.error.HTTPError as e:
        print(f"Sorry, HTTP error: {e.code} in trying to access URL..")
        skip_input = True
    except urllib.error.URLError as e:
        print(f"Sorry, URL error: {e.reason} in trying to access URL.")
        skip_input = True
    except Exception as e:
        print(f"Unexpected {e=}, {type(e)=}")
        skip_input = True

    return webpagetext, skip_input


def generate_response(input,
                      messages=[{"role": "system", "content": SYSMESSAGE}],
                      model=MODEL,
                      temperature=TEMPERATURE,
                      top_p=TOP_P,
                      maxchar=MAXCHAR,
                      debug=DEBUG):
    """Check for input plugins (like webpage urls), retrieve them, submit input
    to GPT model and return reply.

    Parameters:
        input            string   latest user input message
        messages         list     chat history of user & agent messages
                                  in OpenAI's list-of-dicts format like
                                  [{"role":___, "content":___}, {...}, ...]
        model            string   OpenAI's "gpt-4", "gpt-3.5-turbo", etc.
        temperature      float    consistency/creativity parameter 0.0-1.0
        top_p            float    sampling parameter 0.0-1.0
        maxchar          integer  number of chars at which to truncate returned
                                  webapge contents (protecting token count)
        debug            boolean  turn on verbose debug output (or not)
    Returns:
        reply            string   latest chatbot reply message
        metadata         dict     dict of non-msg reply info like token counts
    """

    # Check for possible URL in user-submitted text
    webpagetext = None
    skip_input = False
    url_search = re.search("<<(.*)>>", input, re.IGNORECASE)
    if url_search:
        input = re.sub("<<(.*)>>", "", input)
        url = url_search.group(1)
        webpagetext, skip_input = get_url_contents(url, maxchar)

    if not skip_input:
        reply = None
        metadata = None
        if input:
            messages.append({"role": "user", "content": input})
            if webpagetext is not None:
                messages.append({"role": "user", "content": webpagetext})

        if debug:
            print("Debug output in generate_response() :")
            print(f"messages before submit_to_gpt(): {messages}")
            print(f"model: {model}, temp: {temperature}, top_p: {top_p}")

        try:
            reply, metadata = submit_to_gpt(messages,
                                            model,
                                            temperature,
                                            top_p)
            if debug:
                print(f"messages after submit_to_gpt(): {messages}")

        except Exception as e:
            print(f"Error from submit_to_gpt(): {e}")
            pass

    return reply, metadata, messages


class CmdLineInterpreter(Cmd):
    """OpenAI command line interpreter based on built-in Python Cmd package.
    Various configuration parameters are described in constructor doc-string.
    """

    console = Console()

    def __init__(self,
                 prompt="\n\001\033[01;32m\002😃\001\033[37m\033[01;32m\002 Me:\001\033[00m\002 ",
                 gptprompt="\001\033[01;32m\002🤖\001\033[37m\033[01;36m\002 GPT:\001\033[00m\002 ",
                 code_theme="monokai" if darkdetect.isDark() else "default",
                 intro="Params: <params>\n<instructions>\n",
                 history_file=os.path.expanduser('~/.gpt_history'),
                 sysmessage=SYSMESSAGE,
                 model=MODEL,
                 temperature=TEMPERATURE,
                 top_p=TOP_P,
                 maxchar=MAXCHAR,
                 allow_injections=ALLOW_INJECTIONS,
                 debug=DEBUG,
                 ):
        """
        Parameters:
            prompt           string   user input prompt string in CLI
            gptprompt        string   chatbot response prompt string in CLI
            code_theme       string   syntax highlight theme in Rich in CLI
            intro            string   opening/greeting lines in CLI
            history_file     string   CLI history path (default ~/.gpt_history)
            messages         list     chat history of user & agent messages
                                      in OpenAI's list-of-dicts format like
                                      [{"role":___, "content":___}, {...}, ...]
            model            string   OpenAI's "gpt-4", "gpt-3.5-turbo", etc.
            temperature      float    consistency/creativity parameter 0.0-1.0
            top_p            float    sampling parameter 0.0-1.0
            maxchar          integer  numchars to trunc inserted webpage at
            allow_injections boolean  allow insertion of weblinks
            debug            boolean  turn on verbose debug output (or not)
        """

        self.prompt = prompt
        self.gptprompt = gptprompt
        self.code_theme = code_theme
        self.intro = intro
        self.history_file = history_file
        self.messages = [{"role": "system", "content": sysmessage}]
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.maxchar = maxchar
        self.allow_injections = allow_injections
        self.debug = debug

        # Set up the readline handling:
        if 'libedit' in readline.__doc__:
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        if os.path.exists(self.history_file):
            readline.read_history_file(self.history_file)

        # Initialize the greeting/intro message on startup
        Cmd.__init__(self)
        d = locals()
        self.intro = self.intro.replace(
            '<params>',
            # list the gpt model's params without the client UI params:
            str({k:d[k] for k in d if
                k!='self' and
                k!='prompt' and
                k!='gptprompt' and
                k!='code_theme' and
                k!='intro' and
                k!='history_file' and
                k!='messages'
             })
        )
        self.intro = self.intro.replace(
            '<instructions>',
            "You can enter page contents of a URL by putting "
             "the URL in double chevrons like this: <<URL>> "
        )
        self.cmdloop(intro=self.intro)

    def cmdloop(self, intro):
        try:
            Cmd.cmdloop(self, self.intro)
        except KeyboardInterrupt:
            print()
            self.do_exit()
        except EOFError:
            self.do_exit()

    def emptyline(self):
        pass

    def do_exit(self, line=None):
        print("\n\nOk, goodbye...")

        return True

    do_EOF = do_exit  # enables ctrl-D to exit

    def default(self, line):
        if line == "exit" or line == "quit" or line == "q":
            return self.do_exit()

        # Turn on multi-line entry mode if <<multi>> tag in input line;
        # mode ends with <<end>> tag.
        if "<<multi>>" in line:
            while True:
                line = line.replace("<<multi>>", "")
                next_line = input()
                if "<<end>>" in next_line:
                    line += "\n" + next_line.replace("<<end>>", "").replace("<<multi>>", "")
                    break
                line += "\n" + next_line.replace("<<multi>>", "")

        # Append latest input line just entered to history file
        readline.append_history_file(1, self.history_file)

        if self.debug:
            print(f"messages before generate_response(): {self.messages}")
            print(f"line before generate_response(): {line}")

        reply, metadata, self.messages = generate_response(
            line,
            self.messages,
            self.model,
            self.temperature,
            self.top_p,
            self.maxchar,
            self.debug
        )

        if self.debug:
            print(f"messages after generate_response(): {self.messages}")

        # Handle markdown and syntax highlighting and word/line wrapping;
        # technically could just use print() instead, just not as pretty.
        self.console.print(
            f"[grey78][{metadata['prompt_tokens']} prompt-tokens; "
            "includes resubmission of all history this session plus "
            "page contents of any urls given...][/grey78]")
        self.console.print(" ")
        self.console.print(
            Markdown(self.gptprompt + reply, code_theme=self.code_theme)
        )
        self.console.print(
            f"[grey78][{metadata['completion_tokens']} "
            "completion-tokens just for this response...][/grey78]"
        )
        self.console.print(" ")


def gradio_response(input, history):
    """Format entries for our generate_response() function, to match the form
    expected by gradio.  For use with gradio's ChatInterface class. See
    https://www.gradio.app/docs/chatinterface

    Parameters:
        input    string    latest user input message
        history  list      gradio-formatted history of user & agent messages:
                           a string input message and list of two-element lists
                           of the form [[user_msg_str, bot_msg_str], ...]
                           representing the chat history.
    Returns:
        reply    string    latest chatbot reply message
    """

    if webapp_state["debug"]:
        print("Debug output in gradio_response() :")
        print(f"model: {webapp_state['model']}, "
              f"temp: {webapp_state['temperature']}, "
              f"top_p: {webapp_state['top_p']}")
        print(f"messages before generate_response(): {webapp_state['messages']}")

    reply, metadata, messages = generate_response(
        input,
        messages=webapp_state["messages"],
        model=webapp_state["model"],
        temperature=webapp_state["temperature"],
        top_p=webapp_state["top_p"],
        maxchar=webapp_state["maxchar"],
        debug=webapp_state["debug"]
    )

    return reply


@click.group()
def clickgrp():
    """Group the two entrypoint commands into one command line"""
    pass

def common_options(f):
    """Decorator to avoid redundantly specifying shared/common options"""
    options = [
        click.option('--model', default=MODEL, help='OpenAI model'),
        click.option('--temperature', default=TEMPERATURE, help='Consistency/creativity parameter'),
        click.option('--top_p', default=TOP_P, help='Sampling parameter'),                                                                                                                  click.option('--maxchar', default=20000, help='Number of chars at which to truncate returned webpage contents'),
        click.option('--allow_injections', default=ALLOW_INJECTIONS, help='Allow insertion of weblinks'),
        click.option('--maxchar', default=MAXCHAR, help='Truncate webpage content at this number of characters'),
        click.option('--sysmessage', default=SYSMESSAGE, help='Initial system message specifying chatbot personality/functionality'),
        click.option('--debug', default=DEBUG, help='Enable verbose debug output'),
    ]
    for option in reversed(options):
        f = option(f)
    return f

@clickgrp.command()
@common_options
@click.option('--prompt', default=PROMPT, help='User input prompt string in CLI')
@click.option('--gptprompt', default=GPTPROMPT, help='Chatbot response prompt string in CLI')
@click.option('--code_theme', default=CODE_THEME, help='Syntax highlight theme in Rich in CLI')
@click.option('--intro', default=INTRO, help='Opening/greeting lines in CLI')
@click.option('--history_file', default=HISTORY_FILE, help='CLI history path')
def cli(**kwargs):
    """Command for CLI entry point"""
    CmdLineInterpreter(**kwargs)

@clickgrp.command()
@common_options
def webapp(**kwargs):
    """Command for Gradio-based web app entry point"""
    webapp_state.update({
        "messages": [{"role": "system", "content": kwargs["sysmessage"]}],
        "model": kwargs["model"],
        "temperature": kwargs["temperature"],
        "top_p": kwargs["top_p"],
        "maxchar": kwargs["maxchar"],
        "allow_injections": kwargs["allow_injections"],
        "debug": kwargs["debug"]
    })
    gr.ChatInterface(
        gradio_response,
        title="GPTclient Chatbot",
        analytics_enabled=False,
    ).launch()


if __name__ == '__main__':
    clickgrp()
