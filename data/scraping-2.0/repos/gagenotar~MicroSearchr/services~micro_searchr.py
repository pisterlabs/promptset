from utils.openai_client import OpenAIClient, get_openai_client
from utils.searchr_client import SearchrClient, get_searchr_client
from schemas.micro_searchr import MicroSearchrResponse


class MicroSearchrService:
    def __init__(self, openai_client: OpenAIClient, searchr_client: SearchrClient):
        self.openai_client = openai_client
        self.searchr_client = searchr_client
        self._search_prompt = """
        "Summarize the serper.dev search results array, based on the user's query, in 150 words or less.
        The user's query was: {0}
        """

    # @staticmethod
    def search(self, query: str) -> MicroSearchrResponse:
        # Search query using serper api
        responses = self.searchr_client.lookup(query)

        # send context + query + styling to OpenAIClient
        system_prompt = self._search_prompt.format(query)

        search_summary = self.openai_client.generate_text(
            prompt=responses,
            max_tokens=300,
            system_prompt=system_prompt
        )

        response = MicroSearchrResponse()
        response.message = search_summary.content

        # return response
        print(response)
        return response


def get_micro_searchr_service():
    openai_client: OpenAIClient = get_openai_client()
    searchr_client: SearchrClient = get_searchr_client()
    return MicroSearchrService(openai_client, searchr_client)


if __name__ == '__main__':
    searchr_service = get_micro_searchr_service()
    new_event = searchr_service.search("banana hammock")
    print(new_event)
