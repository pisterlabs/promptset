import logging
import openai
import tiktoken
import Keys

openai.api_key = Keys.OPEN_AI_KEY
enc = tiktoken.get_encoding("cl100k_base")




logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class GPT:
    def __init__(self, System, model="gpt-4"):
        self.Sys_Message = System
        self.model = model
        self.messages = []
    def __str__(self):
        return f"{self.__class__} instance"
    def _get_prompt(self, input=None):
        context = []
        context.append({"role": "system", "content": f"{self.Sys_Message}"})
        if not input is None:
            self.messages.append(input)
        for index, message in enumerate(self.messages):
            if index % 2 == 0:
                context.append({"role": "user", "content": message})
            else:
                context.append({"role": "assistant", "content": message})
        log.debug(f"Context is: {context}")
        return context
    def set_initial_message(self, message: str):
        self.messages = [message]
    def _attempt_chat_completion(self, input=None):
        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=self._get_prompt(input),
            )
            return completion
        except Exception as e:
            log.info(f"Chat completion has failed {e}")

    def create_chat_completion(self, input = None):
        completion = self._attempt_chat_completion(input)
        log.debug(completion.choices[0].message.content)
        self.messages.append(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def memory_wipe(self):
        self.messages = []

