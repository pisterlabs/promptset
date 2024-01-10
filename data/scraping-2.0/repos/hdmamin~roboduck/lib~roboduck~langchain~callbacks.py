"""Langchain callbacks. This allows us to do things like type output
incrementally rather than waiting for a full completion to finish.
"""
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import time

from roboduck.utils import colored


class LiveTypingCallbackHandler(StreamingStdOutCallbackHandler):
    """Streams to stdout by types one character at a time and allows us to
    control text color and typing speed.

    The parent class mostly just prevents us from having to write a
    bunch of boilerplate methods that do nothing, but it does make sense
    since this callback implements a specific subcase of streaming to stdout.
    """

    always_verbose = True

    def __init__(self, color='green', sleep=.01):
        """
        Parameters
        ----------
        color : str
            Color to print gpt response in. Passing in an empty str (or None)
            will use the default color.
        sleep : float
            Time to wait after "typing" each character. Using zero pause is a
            bit annoying to read comfortably, IMO.
        """
        self.color = color
        self.sleep = sleep

    def on_llm_new_token(self, token, **kwargs):
        """Runs on new LLM token. Only called when streaming is enabled.

        Parameters
        ----------
        token : str
        """
        for char in token:
            sys.stdout.write(colored(char, self.color))
            time.sleep(self.sleep)
        sys.stdout.flush()
