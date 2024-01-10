import asyncio
import unittest

from instructor import OpenAISchema
from openai import APIConnectionError, APITimeoutError
from pydantic import Field

from openai_request_runner import run_openai_requests
from openai_request_runner.request_runner import process_api_requests_from_list


# Needs OpenAI API Key in environment variable OPENAI_API_KEY
class TestOpenAIRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.example_input = [{"id": 0, "prompt": "What is 1+1?"}]
        return super().setUp()

    def test_basic_functionality(self):
        try:
            # results = asyncio.run(process_api_requests_from_list(self.example_input))
            results = run_openai_requests(self.example_input)
            assert "2" in results[0]["content"]  # type: ignore
        except (APITimeoutError, APIConnectionError) as e:
            self.skipTest(f"Skipped due to Connection Error: {e}")

    def test_get_responses_function(self):
        class Answer(OpenAISchema):
            """Answer for the user query."""

            answer: int = Field(
                ...,
                description="The answer for the user query (only numerical).",
            )

        def postprocess_response(response, request_json: dict, metadata: dict):
            return Answer.from_response(response)

        try:
            results = asyncio.run(
                process_api_requests_from_list(
                    self.example_input,
                    functions=[Answer.openai_schema],
                    function_call={"name": Answer.openai_schema["name"]},
                    postprocess_function=postprocess_response,
                    max_tokens=10,
                )
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].answer, 2)  # type: ignore

        except (APITimeoutError, APIConnectionError) as e:
            self.skipTest(f"Skipped due to Connection Error: {e}")


if __name__ == "__main__":
    unittest.main()
