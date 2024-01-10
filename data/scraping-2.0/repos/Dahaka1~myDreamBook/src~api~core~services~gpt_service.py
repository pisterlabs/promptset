from __future__ import annotations

from fastapi import Request
import openai

from infra.requesters import GPTRequester
from container import gpt_requester
from utils.constants import DEFAULT_GPT_SYSTEM_MESSAGE
from utils.logging import get_logger
from models import DreamIn, Dream, GPTQuery, GPTMessage

service = None

logger = get_logger()


def get_gpt_service() -> GPTService:
	global service
	if not service:
		service = GPTService(gpt_requester)
	return service


class GPTService:
	def __init__(
		self,
		requester: GPTRequester
	):
		self._requester = requester

	async def get_dream_interpretation(self, dream_query: DreamIn) -> Dream:
		dream_query = self._prepare_dream_query(dream_query)
		try:
			dream_response = await self._requester.request(
				query=dream_query
			)
			dream_interpretation = dream_response.content
		except openai.APIError:
			dream_interpretation = await self._requester.request_on_error(dream_query)
		logger.info(f"User request handled")
		return Dream(
			content=dream_query.content, interpretation=dream_interpretation
		)

	@staticmethod
	def _prepare_dream_query(dream_query: DreamIn) -> GPTQuery:
		query_messages = [{"content": dream_query.content}]
		system_message = GPTMessage.create_system_message(DEFAULT_GPT_SYSTEM_MESSAGE)
		return GPTQuery(messages=[system_message] + query_messages)

