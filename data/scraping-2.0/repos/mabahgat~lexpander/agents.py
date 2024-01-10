import time

import json

import uuid

from typing import List
from abc import ABC, abstractmethod
import logging

from config import global_config
import openai
from requests import Session

from utils import SynchronousThrottle, CachableCall


logging.basicConfig(level=global_config.logging)


class Example:
	def __init__(self, word: str, label: str, context_or_definition: str = None, example: str = None):
		self.word = word
		self.label = label
		self.context_or_definition = context_or_definition
		self.example = example

	def __str__(self):
		return f'{self.word} -> {self.label}'

	def __repr__(self):
		return self.__str__()


# Parent agent class that takes definition, example and word to generate a label from a list of labels
class Agent(ABC):
	def __init__(self,
				 lexicon_name: str,
				 labels: List[str],
				 call_throttle_per_minute: int,
				 cache_file: str,
				 ask_for_label_only: bool = False):
		self._lexicon_name = lexicon_name
		self._labels = [l.lower() for l in labels]
		self._labels_string = ', '.join(self._labels)
		self._logging = logging.getLogger(__name__)

		self._throttled_call = SynchronousThrottle(call_count=call_throttle_per_minute, period_in_minutes=1)
		self._cachable_call = CachableCall(cache_file_path=cache_file)

		self._ask_for_label_only = ask_for_label_only

	# generates a label from a list of labels
	def label(self,
			  word: str,
			  definition_or_context: str,
			  example: str = None) -> str:
		prompt = self.prompt(word, definition_or_context, example)
		agent_answer = self.answer(prompt)
		return self.extract_label(agent_answer)

	def prompt(self,
			   word: str,
			   definition_or_context: str,
			   example: str = None) -> str:
		if example is None:
			prompt = self.prompt_with_context(definition_or_context, word)
		else:
			prompt = self.prompt_with_details(definition_or_context, example, word)
		if self._ask_for_label_only:
			prompt += '\nRespond with label only'
		return prompt

	def prompt_with_context(self,
							context: str,
							word: str) -> str:
		assert (context)
		assert (word)
		prompt = ''
		prompt += 'In the following context: \n'
		prompt += f'"{context}", \n'
		prompt += f'which label should be assigned to the word "{word}" ' \
				  f'from the following "{self._lexicon_name}" labels: {self._labels_string}?'
		self._logging.debug(f'Context generated prompt: {prompt}')
		return prompt

	def prompt_with_details(self,
							definition: str,
							example: str,
							word: str) -> str:
		assert (definition)
		assert (example)
		assert (word)
		prompt = ''
		prompt += 'For a word with the following meaning: \n'
		prompt += f'"{definition}\n"'
		prompt += 'in this example: \n'
		prompt += f'"{example}" \n'
		prompt += f'which label should be assigned to the word "{word}" ' \
				  f'from the following "{self._lexicon_name}" labels: {self._labels_string}?'
		self._logging.debug(f'Details generated prompt: {prompt}')
		return prompt

	def label_with_few_shot(self,
							training_examples: List[Example],
							word: str,
							definition_or_context: str,
							example: str = None) -> str:
		prompt = ''
		prompt += 'Using the following LIWC 2015 labels for the following words as example:\n'
		for training_example in training_examples:
			prompt += f'"{training_example.word}" labeled as "{training_example.label}"\n'
		prompt += self.prompt(word, definition_or_context, example)
		agent_answer = self.answer(prompt)
		return self.extract_label(agent_answer)

	def answer(self, prompt: str) -> str:
		return self._cachable_call(self._throttled_call, self.answer_impl, prompt)

	@abstractmethod
	def answer_impl(self, prompt: str) -> str:
		pass

	# Find which of the labels occurs first in the answer string
	def extract_label(self, answer: str) -> str:
		labels = self.extract_labels(answer)
		if len(labels) == 0:
			return None
		else:
			return self.extract_labels(answer)[0]

	def extract_labels(self, answer: str) -> List[str]:
		answer = answer.lower()
		word_to_index = {}
		for label in self._labels:
			index = answer.find(label)
			if index != -1:
				word_to_index[label] = index
		return sorted(word_to_index, key=word_to_index.get)


class ChatGPT(Agent):
	def __init__(self, lexicon_name: str, labels: List[str], ask_for_label_only: bool = False, max_out_tokens=1024):
		openai.organization = global_config.apis.chat_gpt.org_id
		openai.api_key = global_config.apis.chat_gpt.api_key
		self.model_name = global_config.apis.chat_gpt.model_name
		self.__chat_gpt = openai.Completion()
		self.__max_out_tokens = max_out_tokens

		super().__init__(lexicon_name,
						 labels,
						 call_throttle_per_minute=global_config.apis.chat_gpt.throttle_per_minute,
						 cache_file=global_config.apis.chat_gpt.cache_path,
						 ask_for_label_only=ask_for_label_only)

	def answer_impl(self,
					prompt: str) -> str:
		self._logging.debug(f'engine: {self.model_name}\nprompt: {prompt}\nmax_tokens: {self.__max_out_tokens}')
		response = self.__chat_gpt.create(
			engine=self.model_name,
			prompt=f"{prompt}",
			max_tokens=self.__max_out_tokens,
			temperature=0
		)
		return response['choices'][0]['text']


# Send question to HugChat and return answer
class HugChat(Agent):
	def __init__(self, lexicon_name: str, labels: List[str], ask_for_label_only: bool = False):
		self._chat = HuggingChatApiHelper(user_id=global_config.apis.hugchat.user_id,
										  token=global_config.apis.hugchat.token)
		super().__init__(lexicon_name,
						 labels,
						 call_throttle_per_minute=global_config.apis.hugchat.throttle_per_minute,
						 cache_file=global_config.apis.hugchat.cache_path,
						 ask_for_label_only=ask_for_label_only)

	def answer_impl(self,
					prompt: str) -> str:
		self._chat.new_conversation()
		try:
			response = self._chat.chat(prompt)
		except Exception as e:
			self._logging.error(f'Exception occurred while chatting with Hugging Face: {e}')
			raise e
		finally:
			self._chat.delete_conversation()
		return response


# API interface for Hugging Chat
class HuggingChatApiHelper:
	def __init__(self,
				 user_id: str,
				 token: str,
				 model: str = 'OpenAssistant/oasst-sft-6-llama-30b-xor',
				 temperature=0.9,
				 truncate=1000,
				 max_new_tokens=1024,
				 top_p=0.95,
				 repetition_penalty=1.2,
				 presence_penalty=0.6,
				 top_k=50,
				 return_full_text=False):
		self.user_id = user_id
		self.token = token

		self.model = model
		self.temperature = temperature
		self.truncate = truncate
		self.max_new_tokens = max_new_tokens
		self.top_p = top_p
		self.repetition_penalty = repetition_penalty
		self.presence_penalty = presence_penalty
		self.top_k = top_k
		self.return_full_text = return_full_text

		self.base_url = "https://huggingface.co"

		self.session = Session()

		self.current_conversation_id = None
		self._logging = logging.getLogger(__name__)

	def cookies(self):
		return {
			'hf-chat': self.user_id,
			'token': self.token
		}

	def headers(self) -> dict:
		return {
			'content-type': 'application/json',
		}

	def new_conversation_data(self) -> dict:
		return {
			'model': self.model
		}

	def chat_data(self, prompt: str) -> dict:
		return {
			"inputs": prompt,
			"parameters": {
				"temperature": self.temperature,
				"truncate": self.truncate,
				"max_new_tokens": self.max_new_tokens,
				"stop": [
					"</s>"
				],
				"top_p": self.top_p,
				"repetition_penalty": self.repetition_penalty,
				"top_k": self.top_k,
				"return_full_text": self.return_full_text
			},
			"stream": False,
			"options": {
				"id": str(uuid.uuid4()),
				"is_retry": False,
				"use_cache": False
			}
		}

	def new_conversation(self) -> None:
		response = self.session.post(f"{self.base_url}/chat/conversation",
									 headers=self.headers(),
									 cookies=self.cookies(),
									 json=self.new_conversation_data())
		if response.status_code != 200:
			raise Exception(f'Error creating a new conversation: {response.text}')

		self.current_conversation_id = json.loads(response.text)['conversationId']
		self._logging.debug(f"Created new conversation with id {self.current_conversation_id}")

	def delete_conversation(self) -> None:
		response = self.session.delete(f"{self.base_url}/chat/conversation/{self.current_conversation_id}",
									   cookies=self.cookies(),
									   headers=self.headers())
		if response.status_code != 200:
			raise Exception(f'Error deleting conversation: {response.text}')
		self._logging.debug(f"Deleted conversation with id {self.current_conversation_id}")

	def chat(self, prompt: str):
		retry_count = 20
		retry_delay = 1
		while retry_count > 0:
			response = self.session.post(f"{self.base_url}/chat/conversation/{self.current_conversation_id}",
										 headers=self.headers(),
										 cookies=self.cookies(),
										 json=self.chat_data(prompt))

			if response.status_code != 200:
				self._logging.error(f'Failed with status {response.status_code} and message: {response.text} ... '
									f'retrying in {retry_delay} second '
									f'{retry_count} attempts remaining')
				time.sleep(retry_delay)
				retry_count -= 1
			else:
				return json.loads(response.text)[0]['generated_text']

		raise Exception(f'Failed to get an answer after {retry_count} retries. Last error: {response.text}')