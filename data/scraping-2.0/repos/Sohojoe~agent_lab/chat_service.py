import asyncio
import itertools
import json
import os
import openai
from openai import AsyncOpenAI



class ChatService:
    def __init__(self, api="openai", model_id = "gpt-3.5-turbo"):
        self._api = api
        self._aclient = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self._model_id = model_id

    def _should_we_send_to_voice(self, sentence:str):
        sentence_termination_characters = [".", "?", "!"]
        close_brackets = ['"', ')', ']']

        if sentence is None or sentence.isspace():
            return None

        temination_charicter_present = any(c in sentence for c in sentence_termination_characters)
 
        # early exit if we don't have a termination character
        if not temination_charicter_present:
            return None

        # early exit the last char is a termination character
        if sentence[-1] in sentence_termination_characters:
            return None
        
        # early exit the last char is a close bracket
        if sentence[-1] in close_brackets:
            return None
        
        termination_indices = [sentence.rfind(char) for char in sentence_termination_characters]
        # Filter out termination indices that are not followed by whitespace or end of string
        termination_indices = [i for i in termination_indices if sentence[i+1].isspace()]
        if len(termination_indices) == 0:
            return None
        last_termination_index = max(termination_indices)
        # handle case of close bracket
        while last_termination_index+1 < len(sentence) and sentence[last_termination_index+1] in close_brackets:
            last_termination_index += 1

        text_to_speak = sentence[:last_termination_index+1]
        return text_to_speak
    
    def ignore_sentence(self, text_to_speak):
        # exit if empty, white space or an single breaket
        if text_to_speak.isspace():
            return True
        # exit if not letters or numbers
        has_letters = any(char.isalpha() for char in text_to_speak)
        has_numbers = any(char.isdigit() for char in text_to_speak)
        if not has_letters and not has_numbers:
            return True
        return False

    async def get_responses_as_sentances_async(self, messages, cancel_event=None):
        llm_response = ""
        current_sentence = ""
        delay = 0.1

        while True:
            try:
                response = await self._aclient.chat.completions.create(
                    model=self._model_id,
                    messages=messages,
                    temperature=1.0,  # use 0 for debugging/more deterministic results
                    stream=True
                )

                async for chunk in response:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    chunk_text = chunk.choices[0].delta.content
                    if chunk_text:
                        current_sentence += chunk_text
                        llm_response += chunk_text
                        text_to_speak = self._should_we_send_to_voice(current_sentence)
                        if text_to_speak:
                            current_sentence = current_sentence[len(text_to_speak):]
                            yield text_to_speak, True
                        else:
                            yield current_sentence, False

                if cancel_event is not None and cancel_event.is_set():
                    return
                if len(current_sentence) > 0:
                    yield current_sentence, True
                return

            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2

            # except openai.error.APIConnectionError as e:
            #     print(f"Failed to connect to OpenAI API: {e}")
            #     print(f"Retrying in {delay} seconds...")
            #     await asyncio.sleep(delay)
            #     delay *= 2

            # except openai.error.RateLimitError as e:
            #     print(f"OpenAI API request exceeded rate limit: {e}")
            #     print(f"Retrying in {delay} seconds...")
            #     await asyncio.sleep(delay)
            #     delay *= 2

            except Exception as e:
                print(f"OpenAI API unknown error: {e}")
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2