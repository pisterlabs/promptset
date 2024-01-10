import json
import os
from typing import Any

import cv2
import httpx
import numpy as np
import openai

from fastapi import HTTPException, UploadFile
from httpx import HTTPStatusError

from ..config import settings
from .enum import ContentType
from .utils import (
    convert_image_to_bytes,
    convert_to_json,
    get_azure_vision_headers,
    get_azure_vision_url,
)


# TODO: Implement the functionality to save images to Cloud Storage.


async def process_receipt_url(image_url: str) -> dict[str, Any]:
    """Perform OCR on the image from the provided URL and classify its type using AI."""
    try:
        ocr_results = await perform_ocr({"url": image_url}, ContentType.JSON)
        classification_results = await perform_classification(ocr_results)
        return {"ocr": ocr_results, "result": classification_results}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=400, detail="An error occurred while processing the image."
        )


async def process_receipt_upload(file: UploadFile) -> dict[str, Any]:
    """Perform OCR on the image from the provided URL and classify its type using AI."""
    try:
        img_arr = np.fromstring(await file.read(), np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img_bytes = convert_image_to_bytes(image=img)

        ocr_results = await perform_ocr(img_bytes, ContentType.OCTET_STREAM)
        classification_results = await perform_classification(ocr_results)
        return {"ocr": ocr_results, "result": classification_results}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=400, detail="An error occurred while processing the uploaded image."
        )


async def perform_classification(content: str) -> dict[str, Any]:
    """Perform AI classification on the provided OCR results."""
    try:
        openai.api_key = settings.OPENAI_KEY
        response = openai.ChatCompletion.create(
            model=settings.GPT_MODEL_NAME,
            messages=generate_prompt(content),
        )

        if not response.choices:
            raise Exception("No response from AI.")

        content = response["choices"][0]["message"]["content"]
        return convert_to_json(data_string=content)

    except Exception as e:
        print(f"An error occurred during classification: {e}")
        raise HTTPException(
            status_code=400,
            detail="An error occurred during the AI classification process.",
        )


def generate_prompt(content: str) -> list[dict[str, Any]]:
    """Generate a prompt for OpenAI's GPT model."""
    return [
        {"role": "system", "content": settings.PROMPT_SYSTEM},
        {"role": "assistant", "content": settings.PROMPT_ASSISTANT + content},
        {
            "role": "user",
            "content": settings.PROMPT_USER
            + json.dumps(settings.PROMPT_EXAMPLE, indent=2)
        },
    ]


async def perform_ocr(data: dict | bytes, content_type: ContentType) -> dict[str, Any]:
    """Perform OCR with the provided image."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                get_azure_vision_url(),
                headers=get_azure_vision_headers(content_type),
                params=settings.AZURE_VISION_PARAMS,
                json=data if content_type == ContentType.JSON else None,
                content=data if content_type == ContentType.OCTET_STREAM else None,
            )
        response.raise_for_status()
        return response.json()["readResult"]["content"]

    except HTTPStatusError as e:
        print(f"An error occurred during OCR: {e}")
        raise HTTPException(
            status_code=400, detail="An error occurred during the OCR process."
        )
