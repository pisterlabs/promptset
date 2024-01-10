"""
Code for AI
"""
import asyncio

import dotenv
from openai import AsyncOpenAI
from openai._base_client import AsyncPaginator
from openai.pagination import AsyncPage
from openai.types import FileObject

from chats.utils import write_json_to_logs

dotenv.load_dotenv()


class AIFileSystem:
    def __init__(self, assistant_id: str = None):
        self.client = AsyncOpenAI()

    async def create(self, filename: str, file_bytes: bytes, content_type: str):
        """Upload one file"""
        # Max of 20 files!
        #  ['c', 'cpp', 'csv', 'docx', 'html', 'java', 'json', 'md', 'pdf', 'php',
        #  'pptx', 'py', 'rb', 'tex', 'txt', 'css', 'jpeg', 'jpg', 'js', 'gif',
        #  'png', 'tar', 'ts', 'xlsx', 'xml', 'zip']
        # (filename, file( or bytes), content_type)
        file = (filename, file_bytes, content_type)

        file = await self.client.files.create(
            file=file,
            # "fine-tune" or "assistants"
            purpose="assistants",
        )
        write_json_to_logs(file, "file_create")
        return file

    async def delete(self, file_id: str):
        result = await self.client.files.delete(
            file_id=file_id,
        )
        write_json_to_logs(result, "file_delete")
        return result

    async def retrieve(self, file_id: str):
        result = await self.client.files.retrieve(
            file_id=file_id,
        )
        write_json_to_logs(result, "file_retrieve")
        return result

    async def list(self) -> AsyncPaginator[FileObject, AsyncPage[FileObject]]:
        result = await self.client.files.list()
        write_json_to_logs(result, "file_list")
        return result

    # TODO retrieve contents


if __name__ == "__main__":

    async def main():
        fs = AIFileSystem()

        for key, value in await fs.list():
            if key == "data":
                for file in value:
                    print(file.id)

    asyncio.run(main())
