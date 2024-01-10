from abc import ABC
from typing import override

from langchain.schema import Document


class Responder(ABC):
    """
    base class for responder objects, given an input dictionary it returns only
    the 'answer' field via the produce_response method.
    """

    def __init__(self, response_dict: dict) -> None:
        self._response_dict = response_dict

    def response_dict(self, rd: dict) -> None:
        assert "answer" in rd, "'answer' key not found in response_dict"
        assert isinstance(
            rd["answer"], str
        ), "the content of the response_dict['answer'] is not a string"
        self._response_dict = rd

    def produce_response(self):
        return self._response_dict["answer"]


class ResponderWithScore(Responder):
    """
    class to create responder objects that return both the 'answer' and 'scores' field of the input dictionary,
    via the produce_response method.
    """

    response_keys = ["answer", "scores"]

    # check override
    @override
    def response_dict(self, rd: dict) -> None:
        assert "answer" in rd, "'answer' key not found in response_dict"
        assert isinstance(
            rd["answer"], str
        ), "the content of the response_dict['answer'] is not a string"
        assert "scores" in rd, "'scores' key not found in response_dict"
        assert isinstance(
            rd["scores"], list
        ), "the content of the response_dict['scores'] is not a list"
        assert all(
            isinstance(entry, (float, int)) for entry in rd["scores"]
        ), "the entries of the response_dict['scores'] list are not numbers (float or int)"
        self._response_dict = rd

    def produce_response(self) -> dict:
        return {k: self._response_dict[k] for k in self.response_keys}


class ResponderWithDocument(Responder):
    """
    class to create responder objects that return both the 'answer' and 'documents' field of the input dictionary,
    via the produce_response method.
    """

    response_keys = ["answer", "documents"]

    @override
    def response_dict(self, rd: dict) -> None:
        assert "answer" in rd, "'answer' key not found in response_dict"
        assert isinstance(
            rd["answer"], str
        ), "the content of the response_dict['answer'] is not a string"
        assert "documents" in rd, "'documents' key not found in response_dict"
        assert isinstance(
            rd["documents"], list
        ), "the content of the response_dict['documents'] is not a list"
        assert all(
            isinstance(entry, Document) for entry in rd["documents"]
        ), "the entries of the response_dict['documents'] list are not Langchain's Documents"
        self._response_dict = rd
        self._response_dict["documents"] = [
            {"page_content": str(doc.page_content), "metadata": str(doc.metadata)}
            for doc in rd["documents"]
        ]

    def produce_response(self) -> dict:
        return {k: self._response_dict[k] for k in self.response_keys}
