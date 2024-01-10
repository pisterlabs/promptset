from typing import Any, Dict, List
import openai

from config import (
    CHUNK_SIZE,
    COMMIT_LANGUAGE,
    CONTENTS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


class CommitMessageGenerator:
    def __init__(
        self,
    ) -> None:
        openai.api_key = OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self.chunk_size = CHUNK_SIZE
        self.content = CONTENTS.get(COMMIT_LANGUAGE, CONTENTS["en"])

    def generate_commit_message(self, diff: str) -> str:
        """Generates a commit message based on the diff provided."""
        messages = self._build_messages(diff)
        return self._get_response_from_openai(messages)

    def _build_messages(self, diff: str) -> List[Dict[str, Any]]:
        """Builds a list of messages to send to the OpenAI API."""
        messages = [{"role": "system", "content": self.content["instruction_request"]}]

        content = self._get_processed_diff(diff)
        content += self.content["convention_request"] + self.content["answer_language"]
        messages.append({"role": "user", "content": content})

        return messages

    def _get_processed_diff(self, diff: str) -> str:
        """Processes the diff for OpenAI API input."""
        if len(diff) > self.chunk_size:
            return self.content["commit_msg_request"] + self._summarize_diff(diff)
        return self.content["commit_msg_request"] + diff

    def _summarize_diff(self, diff: str) -> str:
        """Summarizes the diff by processing it in chunks."""
        max_size = self.chunk_size * 5
        summaries = [
            self._get_response_from_openai(
                [
                    {
                        "role": "user",
                        "content": self.content["summarize_request"]
                        + diff[i : i + self.chunk_size],
                    }
                ]
            )
            for i in range(0, len(diff[:max_size]), self.chunk_size)
        ]
        return " ".join(summaries)

    def _get_response_from_openai(self, messages: List[Dict[str, Any]]) -> str:
        """Fetches response from OpenAI API."""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=500,
            temperature=0.5,
            top_p=0.5,
        )
        return response.choices[0].message.content


openai.api_key = OPENAI_API_KEY
