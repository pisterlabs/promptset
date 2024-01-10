from dynamodm import DynaModel  # type: ignore
from fastapi import UploadFile
from glob_utils._decorators import setup_logging  # type: ignore
from openai.types.file_object import FileObject
from pydantic import BaseModel, Field  # pylint: disable=E0611

from ._service import Service
from .assistant import StorageBucket

logger = setup_logging(__name__)


class FileObjectRequest(BaseModel):
    file: UploadFile = Field(
        default=..., description="The file to use for the file object"
    )


class FileObjectResponse(FileObject):
    url: str = Field(default=..., description="The URL of the file object")


class FileObjectModel(DynaModel):
    key: str = Field(..., pk=True)
    sort: str = Field(..., sk=True)
    data: FileObjectResponse


class FileObjectService(Service[FileObjectRequest, FileObjectResponse]):
    @property
    def client(self):
        return self.api.files

    @property
    def storage(self):
        return StorageBucket()

    async def create_(self, *, key: str, data: FileObjectRequest) -> FileObjectResponse:
        file_obj = await self.client.create(file=data.file.file, purpose="assistants")
        filename = data.file.filename or f"{file_obj.id}.bin"
        content_type = data.file.content_type
        body = await data.file.read()
        url = await self.storage.upload(
            body=body,
            key=f"{key}/{file_obj.id}/{filename}",
            content_type=content_type or "application/octet-stream",
        )
        instance = FileObjectResponse(**file_obj.dict(), url=url)
        repository = FileObjectModel(key=key, sort=file_obj.id, data=instance)
        response = await repository.put()
        return response.data

    async def get_(self, *, key: str, sort: str):
        response = await FileObjectModel.get(pk=key, sk=sort)
        return response.data

    async def update_(self, *, key: str, sort: str, data: FileObjectRequest):
        return await self.create_(key=key, data=data)  # type: ignore

    async def delete_(self, *, key: str, sort: str) -> None:
        await self.client.delete(file_id=sort)
        await self.storage.delete(key=key)
        await FileObjectModel.delete(pk=key, sk=sort)

    async def list_(self, *, key: str) -> list[FileObjectResponse]:
        response = await FileObjectModel.query(pk=key)
        return [item.data for item in response]
