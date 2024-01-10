# chat_manager.py
import ast
import logging
from time import sleep

import openai
from openai.types.chat import ChatCompletion, chat_completion_chunk
import config
from file_manager import FileManager


def extract_trans_text(content) -> list:
    speech = []
    for i in content:
        speech.append(i["text"])
    return speech


def extract_resp_content(resp) -> str:
    reply = resp.choices[0].message.content
    return reply


def sim_typing_output(text: str, delay: float = 0.01):
    # a function to simulate typing in  the terminal output
    for i in text:
        # print each character individually with a small delay in between.
        print(i, end="", flush=True)
        sleep(delay)


def chat_completion_to_dict(response):
    # Assuming 'response' is an instance of ChatCompletion
    # Convert it to a dictionary format
    chat_dict = {
        "id": response.id,  # ID of the completion
        "model": response.model,  # Model used for the completion
        "created": response.created,  # Timestamp of creation
        "object": response.object,  # Object type
        "choices": [
            {
                "finish_reason": choice.finish_reason,  # Reason why the generation was stopped
                "index": choice.index,  # Index of the choice
                "message": {
                    "content": choice.message.content,  # Content of the message
                    "role": choice.message.role,  # Role (user or assistant)
                },
                "logprobs": choice.logprobs,  # Log probabilities, if available
            }
            for choice in response.choices
        ],
        "created": response.created,  # Timestamp of creation
        "id": response.id,  # ID of the completion
        "model": response.model,  # Model used for the completion
        "object": response.object,  # Object type
        "usage": {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
    return chat_dict


class ChatSession:
    error_completion = {
        "choices": [
            {
                "message": {
                    "role": "system",
                    "content": "An error occurred with the API request",
                }
            }
        ]
    }

    def __init__(
        self,
        initial_prompt: str = None,
        model: str = None,
        temperature: float = None,
        presence_penalty: float = None,
        token_limit: int = None,
        limit_thresh: float = None,
    ):
        self.ai = openai
        self.ai.api_key = FileManager.read_file("api_keys/keys")
        self.ai.organization = "org-9YiPG54UMFObNmQ2TMOnPCar"
        self.model = model
        if model is None:
            self.model = "gpt-3.5-1106"
        self.transcription_directory = config.TRANSCRIPTION_PATH
        self.initial_prompt = initial_prompt
        if initial_prompt is None:
            self.initial_prompt = "You are a helpful assistant"
        self.messages: list = [{"role": "system", "content": self.initial_prompt}]
        # self.messages: list = json.loads(FileManager.read_file("conversations/conversation_02-08-2023_10-06-14.json"))
        self.temperature = temperature
        if temperature is None:
            self.temperature = 0
        self.presence_penalty = presence_penalty
        if presence_penalty is None:
            self.presence_penalty = 1
        self.model_token_limit = token_limit
        if token_limit is None:
            self.model_token_limit = 16385
        self.limit_thresh = limit_thresh
        if limit_thresh is None:
            self.limit_thresh = 0.4

    def add_user_entry(self, trans):
        speech = extract_trans_text(trans)
        for i in speech:
            if i is not None or "":
                self.messages.append(
                    {"role": "user", "content": i},
                )
            else:
                pass

    def is_model_near_limit_thresh(self, response: ChatCompletion) -> bool:
        if response.usage.total_tokens > self.model_token_limit * self.limit_thresh:
            return True
        else:
            return False

    def add_reply_entry(self, resp):
        reply = extract_resp_content(resp)
        if reply is not None or "":
            self.messages.append(
                {"role": "assistant", "content": reply},
            )
        else:
            pass

    def extract_streamed_resp_deltas(self, resp):
        # this is messy. Could be improved with the use of a thread pool and a Queue
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in resp:
            chunk_message = chunk.choices[0].delta
            if chunk_message.content is not None:
                print(chunk_message.content, end="", flush=True)
            collected_messages.append(chunk_message)
            collected_chunks.append(chunk)
        print("\n\n")
        full_reply_content = "".join([m.get("content", "") for m in collected_messages])
        self.messages.append({"role": "assistant", "content": full_reply_content})
        tstamp = FileManager.get_datetime_string()
        FileManager.save_json(
            f"{config.RESPONSE_LOG_PATH}response_{tstamp}.json", collected_chunks
        )

    def send_request(self):
        try:
            chat_completion = self.ai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                stream=config.STREAM_RESPONSE,
            )
            return chat_completion

        except Exception as e:
            logging.error(f"Error sending request: {e}")
            return self.error_completion

    def summarize_conversation(self):
        print("\nSummarizing conversation. Please wait...\n")
        m = self.messages[1:]
        prompt = [
            {
                "role": "system",
                "content": "You will be receive an json object which contains a conversation. The object has the "
                "following structure:\n [{'role': 'user', 'content': '*content goes here*'}, "
                "{'role': 'assistant', 'content': '*content goes here*'}]\n"
                "The 'role' key defines a speaker in the conversation, while the 'content' key defines "
                "what that speaker said. Your task is to summarize this conversation to less than ten "
                "percent of it's original length and return your summary in the same structure "
                "as the original object:\n"
                "[{'role': 'user', 'content': '*content goes here*'}, "
                "{'role': 'assistant', 'content': '*content goes here*'}]",
            },
            {"role": "user", "content": f"{m}"},
        ]
        try:
            chat_completion = self.ai.chat.completions.create(
                model=self.model, messages=prompt
            )
            return chat_completion
        except Exception as e:
            print(f"Error sending request: {e}")
            return self.error_completion

    def add_summary(self, summary):
        m = [self.messages[0]]
        r: str = summary.choices[0].message.content
        try:
            r_parsed = ast.literal_eval(r)
            for i in r_parsed:
                m.append(i)
        except SyntaxError as e:
            print(f"{e}")

        print("\nSummary complete\n")
        # print(f"\n{self.messages}")
