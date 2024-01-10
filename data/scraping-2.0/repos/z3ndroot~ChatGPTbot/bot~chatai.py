import json
import logging

import openai
import soundfile as sf
import tiktoken
from openai.error import InvalidRequestError
import os


class GPT:
    def __init__(self, config: dict):
        openai.api_key = config["token_openai"]
        openai.proxy = config['proxy']
        self._config = config

    async def create_chat(self, message: str, chat_id: str):
        """
        Gets a full response from the GPT model.
        :param message: Message from user
        :param chat_id: Telegram chat id
        :return: The answer from the model
        """
        response = await self._generate_gpt_response(message, chat_id, stream=False)
        answer = response.choices[0]['message']['content'].strip()
        history = self.__read_file(chat_id)['history']
        history.append({"role": "assistant", "content": answer})
        self.__add_to_history(history, chat_id)

        return answer

    async def create_chat_stream(self, message: str, chat_id: str):
        """
        Stream response from the GTP model
        :param message: Message from user
        :param chat_id: Telegram chat id
        :return: The answer from the model or 'not_finished'
        """
        response = await self._generate_gpt_response(message, chat_id, stream=True)
        answer = ''
        async for item in response:
            if 'choices' not in item or len(item.choices) == 0:
                continue
            delta = item.choices[0].delta
            if 'content' in delta:
                answer += delta.content
                yield answer, False
        answer = answer.strip()
        history = self.__read_file(chat_id)['history']
        history.append({"role": "assistant", "content": answer})
        self.__add_to_history(history, chat_id)
        yield answer, True

    async def _generate_gpt_response(self, message, chat_id, stream=True):
        """
        Request a response from the GPT model
        :param message: The message to send to the model
        :return: The response from the model
        """
        history = self.__read_file(chat_id)['history']
        history.append({"role": "user", "content": message})
        token_len = self.num_tokens_from_messages(history)
        if token_len + self._config['max_tokens'] > self._config['max_all_tokens']:
            logging.warning(
                f"This model's maximum context length is 4097 tokens."
                f" {token_len} in the messages, 1200 in the completion")
            summarize = await self.__summarise(history[:-1])
            self.clear_history(chat_id)
            history = self.__read_file(chat_id)['history']
            history.append({"role": "assistant", "content": summarize})
            history.append({"role": "user", "content": message})
        self.__add_to_history(history, chat_id)

        return await openai.ChatCompletion.acreate(
            model=self._config["model"],
            messages=history,
            temperature=self._config["temperature"],
            max_tokens=self._config["max_tokens"],
            n=1,
            presence_penalty=self._config["presence_penalty"],
            frequency_penalty=self._config["frequency_penalty"],
            stream=stream)

    async def generate_image(self, prompt: str):
        """
        Generates images with DALL·E on prompt
        :param prompt: The prompt to send to the model
        :return: The image URL
        """
        try:
            response = await openai.Image.acreate(
                prompt=prompt,
                n=1,
                size=self._config["image_size"]
            )
            image_url = response['data'][0]['url']
            return image_url
        except InvalidRequestError as e:
            return e.user_message

    async def transcriptions(self, chat_id: str):
        """
        Transcribes the audio file using the Whisper model.
        :param chat_id: Telegram chat id
        :return: Text decoding of audio
        """
        with open(f"audio/{chat_id}.wav", "rb") as audio:
            result = await openai.Audio.atranscribe("whisper-1", audio)

            return result.text

    async def convert_audio(self, chat_id):
        """
        Converts the .ogg file format to .wav
        """
        ogg_file = f"audio/{chat_id}.ogg"
        wav_file = f"audio/{chat_id}.wav"

        data, samplerate = sf.read(ogg_file)
        sf.write(wav_file, data, samplerate)

    async def delete_audio(self, chat_id):
        os.remove(f"audio/{chat_id}.ogg")
        os.remove(f"audio/{chat_id}.wav")

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0301"):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == self._config["model"]:  # note: future models may deviate from this
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens

    async def __summarise(self, conversation) -> str:
        """
        Summarises the conversation history.
        :param conversation: The conversation history
        :return: The summary
        """
        messages = [
            {"role": "assistant", "content": "Summarize this conversation in 700 characters or less"},
            {"role": "user", "content": str(conversation)}
        ]
        response = await openai.ChatCompletion.acreate(
            model=self._config["model"],
            messages=messages,
            temperature=0.4
        )
        return response.choices[0]['message']['content']

    def __write_to_file(self, data, chat_id: str):
        """
        Writing to the history file
        :param data: Data with chat history
        :param chat_id: Telegram chat id
        """
        with open(f'history/{chat_id}.json', "w", encoding="UTF8") as file:
            json.dump(data, file, indent=4)

    def __read_file(self, chat_id: str):
        """
        Read history file
        :param chat_id: Telegram chat id
        :return: list with chat history
        """
        with open(f'history/{chat_id}.json', "r", encoding="UTF8") as file:
            return json.load(file)

    def create_user_history(self, chat_id, username):
        """
        Creating a history file for a new user
        :param chat_id: Telegram chat id
        :param username: Telegram username
        """
        if not os.path.isfile(f'history/{chat_id}.json'):
            self.__write_to_file({
                'username': username,
                'history': [{"role": "system", "content": "You are a helpful assistant."}],
            }, chat_id)
            logging.info(f"A history file was created for a user {username} (id: {chat_id})")

    def __add_to_history(self, history: list, chat_id):
        """
        Adding a prompt and response from a model in the history file
        :param history: chat history list
        :param chat_id: Telegram chat id
        """
        result = self.__read_file(chat_id)
        result['history'] = history
        self.__write_to_file(result, chat_id)

    def system_message(self, message: str, chat_id):
        """
        Changing the system role message
        :param message: system role message
        :param chat_id: Telegram chat id
        """
        result = self.__read_file(chat_id)
        result['history'][0].update({"content": message})
        self.__write_to_file(result, chat_id)

    def clear_history(self, chat_id: str):
        """
        Cleaning the chat history file
        :param chat_id: Telegram chat id
        """
        result = self.__read_file(chat_id)
        result['history'] = [{"role": "system", "content": ""}]
        self.__write_to_file(result, chat_id)
