import g4f
from openai import AsyncOpenAI
from httpx import Timeout

from models import GPTQuery, GPTResponse
from utils.logging import get_logger
from utils.web import retry_on_exc

logger = get_logger()


class GPTRequester:
	_OPENAI_API_TIMEOUT = Timeout(3.0, connect=2.0)

	def __init__(
		self,
		client: AsyncOpenAI,
	):
		self._client = client
		self._g4f_client = g4f

		self._g4f_client.debug.logging = False

	async def request(self, query: GPTQuery) -> GPTResponse:
		response = await self._client.chat.completions.create(
			**query.jsonable_dict(), timeout=self._OPENAI_API_TIMEOUT
		)
		return GPTResponse(**response.model_dump())

	async def request_on_error(self, query: GPTQuery) -> str:
		result = await self._handle_request_by_third_party_api(query)
		return result

	@retry_on_exc([RuntimeError])
	async def _handle_request_by_third_party_api(self, query: GPTQuery) -> str:
		query_data = query.jsonable_dict()
		query_data["model"] = g4f.models.default
		g4f_response = self._g4f_client.ChatCompletion.create(
			**query_data
		)
		return g4f_response
