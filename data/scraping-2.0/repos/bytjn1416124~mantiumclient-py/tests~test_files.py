from unittest import mock

from mantiumapi import OpenAIFiles

from mantiumapi.files import OpenAIFile

FILES = {
    "data": {
        "id": "mantium-1",
        "type": "openai_file",
        "attributes": {
            "organization": "mantium-1",
            "files": [
                {
                    "object": "file",
                    "id": "file-lNS3HaiyQB9wWHu1r7mWk703",
                    "purpose": "classifications",
                    "filename": "bank77 (1) (4).json",
                    "bytes": 1077073,
                    "created_at": 1638183467,
                    "status": "processed",
                    "status_details": 'null'
                }
            ],
            "size": 1
        },
        "relationships": {}
    },
    "included": [],
    "meta": {},
    "links": {}
}


def mocked_requests(*args, **kwargs):
    class MockResponse:
        def __init__(self, data, status_code):
            self.content = data
            self.json_data = data
            self.status_code = status_code

        def json(self):
            return self.json_data

    if args[0] == 'GET' and args[1] == 'https://api.mantiumai.com/v1/files/openai_files':
        if kwargs['params']['file_type'] == 'FILES_ONLY':
            return MockResponse(FILES, 200)


@mock.patch(
    'jsonapi_requests.request_factory.requests.request', side_effect=mocked_requests
)
def test_files(mock_get):
    target = OpenAIFiles.get_list()
    assert isinstance(target[0], OpenAIFile)
    assert target[0].id == 'file-lNS3HaiyQB9wWHu1r7mWk703'
    assert target[0].purpose == 'classifications'
    assert target[0].filename == 'bank77 (1) (4).json'
    assert target[0].bytes == 1077073
    assert target[0].created_at == 1638183467
    assert target[0].status == 'processed'
    assert target[0].status_details == 'null'
