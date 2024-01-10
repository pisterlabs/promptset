from openai.types.file_deleted import FileDeleted
from typing import Literal
from openai.types.file_list_params import FileListParams
from openai.types.file_object import FileObject
from openai import OpenAI
# For the retrival we will upload files and in message we will pass the file id.
# Remember to destroy the file afterwards

# For Cost Optimization Create a Thread with Attached File Once and then Use it for All Operations/Questions Related to That File


class TravelFiles():
    def __init__(self, client: OpenAI):
        if (client is None):
            raise Exception("OpenAI Client is not initialized")
        self.client = client

    def list_files(self, purpose: str = 'assistants') -> FileListParams:
        """Retrieve a list of files with the specified purpose."""
        files = self.client.files.list(purpose=purpose)
        file_list = files.model_dump()
        return file_list['data'] if 'data' in file_list else []

    def upload_file(self, file_path: str, purpose: Literal['fine-tune', 'assistants'] = 'assistants') -> str:
        """Create or find a file in OpenAI. 
        https://platform.openai.com/docs/api-reference/files/list
        Returns File ID. """

        with open(file_path, "rb") as file:
            file_obj: FileObject = self.client.files.create(
                file=file, purpose=purpose)
            self.file_id: str = file_obj.id
            return file_obj.id

    def deleteFile(self, file_id: str) -> dict[str, FileDeleted | str]:
        """Delete an Uploaded File
        args: Pass file Id
        returns deleted file id, object and deleted (bool)"""

        response: dict[str, FileDeleted | str] = {}
        try:
            deleted_file = self.client.files.delete(
                file_id)  # Assuming this returns FileDeleted
            response['data'] = deleted_file
            response['status'] = 'success'
            print("Deleted File", response['data'])

        except Exception as e:
            # Handle other potential exceptions
            response['status'] = 'error'
            response['error'] = str(e)

        return response
