from typing import Literal

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from openai.types.file_object import FileObject

app = APIRouter()
ai = AsyncOpenAI()


@app.post("/api/file", response_model=FileObject)
async def upload_file(
    file: UploadFile = File(...),
    purpose: Literal["assistants", "fine-tune"] = "assistants",
):
    """
        Uploads all files.

        Example response:
        {
      "data": [
        {
          "id": "file-GaJEbGcpm1SNNZwGQwPS0r4n",
          "bytes": 6976,
          "created_at": 1699590534,
          "filename": "upload",
          "object": "file",
          "purpose": "assistants",
          "status": "processed",
          "status_details": null
        },
        {
          "id": "file-vJjootMMgLc2IlY9ZFRRWi6d",
          "bytes": 1118,
          "created_at": 1699563454,
          "filename": "upload",
          "object": "file",
          "purpose": "assistants",
          "status": "processed",
          "status_details": null
        },
        {
          "id": "file-zMAzCWp7pwDCZ1G4m8SlG3Q1",
          "bytes": 39625,
          "created_at": 1699451500,
          "filename": "None",
          "object": "file",
          "purpose": "assistants",
          "status": "processed",
          "status_details": null
        }
      ],
      "object": "list",
      "has_more": false
    }
    """
    file_content = await file.read()
    response = await ai.files.create(file=file_content, purpose=purpose)
    return response


@app.get("/api/file", response_class=StreamingResponse)
async def get_files(purpose: Literal["assistants", "fine-tune"] = "assistants"):
    """
        Returns a file.

        Example response:

      "id": "file-7FxeGCi9ic6RFoodV1IIGKIj",
      "bytes": 6976,
      "created_at": 1699590689,
      "filename": "upload",
      "object": "file",
      "purpose": "assistants",
      "status": "processed",
      "status_details": null
    }
    """
    response = await ai.files.list(purpose=purpose)

    async def generator():
        async for file in response:
            yield f"data: {file.json()}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.delete("/api/file/{file_id}", response_model=None)
async def delete_file(file_id: str):
    """
    Deletes a file.
    """
    await ai.files.delete(file_id=file_id)


@app.get("/api/file/{file_id}", response_model=FileObject)
async def retrieve_files(file_id: str):
    """
        Returns a file.

        Example response:


      "id": "file-7FxeGCi9ic6RFoodV1IIGKIj",
      "bytes": 6976,
      "created_at": 1699590689,
      "filename": "upload",
      "object": "file",
      "purpose": "assistants",
      "status": "processed",
      "status_details": null
    }
    """
    response = await ai.files.retrieve(file_id=file_id)
    return response
