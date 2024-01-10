#   Copyright [2023] [Sunholo ApS]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from typing import Any, Dict, List, Union
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult

import threading
import re
import logging
logging.basicConfig(level=logging.INFO)

class ContentBuffer:
    def __init__(self):
        self.content = ""
        logging.debug("Content buffer initialized")
    
    def write(self, text: str):
        self.content += text
        logging.debug(f"Written {text} to buffer")
    
    def read(self) -> str:
        logging.debug(f"Read content from buffer")    
        return self.content

    def clear(self):
        logging.debug(f"Clearing content buffer")
        self.content = ""
    

class BufferStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, content_buffer: ContentBuffer, tokens: str = ".?!\n", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_buffer = content_buffer

        self.tokens = tokens
        self.buffer = ""
        self.stream_finished = threading.Event()
        self.in_code_block = False
        self.in_question_block = False
        self.question_buffer = ""
        logging.info("Starting to stream LLM")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        logging.debug(f"on_llm_new_token: {token}")

        self.buffer += token

        # Toggle the code block flag if the delimiter is encountered
        if '```' in token:
            self.in_code_block = not self.in_code_block

        # Process the buffer if not inside a code block
        if not self.in_code_block:
            self._process_buffer()

    def _process_buffer(self):

        # Check for the last occurrence of a newline followed by a numbered list pattern
        matches = list(re.finditer(r'\n(\d+\.\s)', self.buffer))
        if matches:
            # If found, write up to the start of the last match, and leave the rest in the buffer
            last_match = matches[-1]
            start_of_last_match = last_match.start() + 1  # Include the newline in the split
            self.content_buffer.write(self.buffer[:start_of_last_match])
            self.buffer = self.buffer[start_of_last_match:]
        else:
            # If not found, and the buffer ends with one of the specified ending tokens, write the entire buffer
            if any(self.buffer.endswith(t) for t in self.tokens):
                self.content_buffer.write(self.buffer)
                self.buffer = ""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:

        if self.buffer:
            # Process the remaining buffer content
            self.content_buffer.write(self.buffer)
            self.buffer = "" # Clear the remaining buffer
            logging.info("Flushing remaining LLM response buffer")

        self.stream_finished.set() # Set the flag to signal that the stream has finished
        logging.info("Streaming LLM response ended successfully")