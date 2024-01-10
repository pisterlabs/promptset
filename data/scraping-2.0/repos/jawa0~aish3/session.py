import logging

from openai import OpenAI, chat
import os
from queue import Queue
from typing import Callable, Dict, List, Optional

from audio_service import AudioService


class ChatCompletionHandler:
    def __init__(self, 
                 start_handler: Optional[Callable[[], None]]=None,
                 chunk_handler: Optional[Callable[[str], None]]=None, 
                 done_handler: Optional[Callable[[], None]]=None):
        
        self._start_handler = start_handler
        self._chunk_handler = chunk_handler
        self._done_handler = done_handler

        self._started = False
        self._done = False
    

    def on_start(self) -> None:
        if self._start_handler is not None:
            self._start_handler()
        self._started = True

            
    def on_text_chunk(self, text: str) -> None:
        if self._chunk_handler is not None:
            self._chunk_handler(text)


    # @todo: accumulate text and send to done?
    def on_done(self) -> None:
        if self._done_handler is not None:
            self._done_handler()
        self._done = True


class Session:
    def __init__(self):
        logging.debug("Client Session.__init__")

        
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
                                    # organization=os.getenv("OPENAI_ORGANIZATION"))

        self._running_completions: Dict[chat.completion, List[ChatCompletionHandler]] = {}
        
        self._audio = AudioService()
        self._channels = {}

        self.gui = None


    def start(self):
        logging.debug("Client Session.start")
        self._audio.start()

    
    def stop(self):
        logging.debug("Client Session.stop")
        self._audio.stop()


    def update(self):
        # logging.debug("ENTER Client Session.update")
        done_completions = []

        # Pump chat completions...
        for completion in self._running_completions:
            try:
                chunk = next(completion)    # Could raise StopIteration
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content'):
                    chunk_text = chunk.choices[0].delta.content
                    for handler in self._running_completions[completion]:
                        if hasattr(handler, 'on_text_chunk'):
                            handler.on_text_chunk(chunk_text)

            except StopIteration:
                # This means the completion we tried to call next() on is done.
                # 1. Call all done handlers for that completion
                for handler in self._running_completions[completion]:
                    if hasattr(handler, 'on_done'):
                        handler.on_done()

                # 2. Mark that completion for removal, but since we're looping
                #    over the dict, we can't remove it yet.

                done_completions.append(completion)
                continue

        # Remove finished completion callbacks
        for done_completion in done_completions:
            del self._running_completions[done_completion]
        
        # logging.debug("EXIT Client Session.update")


    def publish(self, channel_name, obj):
        if channel_name in self._channels:
            for q in self._channels[channel_name]:
                q.put(obj)


    def subscribe(self, channel_name: str) -> Queue:
        q = Queue()
        if channel_name not in self._channels:
            self._channels[channel_name] = []
        self._channels[channel_name].append(q)
        return q



    def llm_send_streaming_chat_request(self, model, chat_messages, handlers: List[ChatCompletionHandler]=[]):
        assert(model == 'gpt-4' or model == 'gpt-3.5-turbo' or model == 'gpt-4-1106-preview')
        
        completion = self.openai_client.chat.completions.create(model=model, messages=chat_messages, stream=True)
        logging.debug(chat_messages)
        self._running_completions[completion] = handlers

        for handler in handlers:
            if hasattr(handler, 'on_start'):
                handler.on_start()
