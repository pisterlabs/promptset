from typing import Any, List, Optional

import openai
from aiofauna import *
from pydantic import Field

from ..schemas import FunctionDocument


async def chat_completion(text: str, context: Optional[str] = None) -> str:
	if context is not None:
		messages = [
			{"role": "user", "content": text},
			{"role": "system", "content": context},
		]
	else:
		messages = [{"role": "user", "content": text}]
	response = await openai.ChatCompletion.acreate(
		model="gpt-3.5-turbo-16k-0613", messages=messages
	)
	return response["choices"][0]["message"]["content"]


class Quiz(FunctionDocument):
	"""Generates a set of questions of a given topic."""

	topic: str = Field(description="Topic to generate questions about.")
	quantity: int = Field(
		default=5, gt=0, lt=11, description="Number of questions to generate."
	)
	questions: Optional[List[str]] = Field(
		default=None, description="List of questions to generate answers for."
	)

	async def ask(self, **kwargs: Any) -> str:
		context = f"You are an expert on {self.topic}."
		text = f"Please formulate a non trivial question about {self.topic} to asses candidate knowledge about the subject. These questions were already asked: {self.questions}, ask a different one."
		response = await chat_completion(text, context=context)
		if self.questions is None:
			self.questions = [response]
		else:
			self.questions.append(response)
		return response

	@process_time
	@handle_errors
	async def run(self, **kwargs: Any) -> List[str]:
		for _ in range(self.quantity):
			await self.ask(**kwargs)
			return self.questions


class Song(FunctionDocument):
	"""Generates a song of a given genre."""

	title: str = Field(description="Title of the song.")
	genre: str = Field(default="pop", description="Genre of the song.")
	lyrics: Optional[str] = Field(default=None, description="Lyrics of the song.")

	async def run(self, **kwargs: Any) -> str:
		context = f"You are a songwriter. You are writing a {self.genre} song called {self.title}."
		text = f"Generate lyrics for the song {self.title}."
		response = await chat_completion(text, context=context)
		self.lyrics = response
		return self


class Blog(FunctionDocument):
	"""Generates a blog post of a given topic."""

	topic: str = Field(description="Topic of the blog post.")
	title: str = Field(description="Title of the blog post.")
	content: Optional[str] = Field(
		default=None, description="Content of the blog post."
	)

	async def run(self, **kwargs: Any) -> str:
		context = f"You are a blogger. You are writing a blog post about {self.topic}."
		text = f"Generate content for the blog post {self.title}."
		response = await chat_completion(text, context=context)
		self.content = response
		return self
