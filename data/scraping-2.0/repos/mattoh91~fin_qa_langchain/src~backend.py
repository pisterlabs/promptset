import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

sys.path.append("./src")
import general_utils
from langchain_utils import (
    get_conversation_chain,
    get_pdf_text,
    get_text_chunks,
    get_vectorstore,
)

# Constants
CONV_CHAIN = None
CWD = Path.cwd().resolve()
LOG_CONFIG = CWD / "conf/base/logging.yaml"
VECTORSTORE = None

# Pydantic models
class Data(BaseModel):
    question: str
    temperature: float


# Initialise fastapi app
app = FastAPI()

# Setup logger
logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
general_utils.setup_logging(logging_config_path=LOG_CONFIG)

# Route requests to root over to swagger UI
@app.get("/")
def redirect_swagger():
    response = RedirectResponse("/docs")
    return response


# Upload
# Side note: If request body has multiple files and data, need to specify data with Form()
@app.post("/upload")
async def upload(
    files: Annotated[list[UploadFile], File()], openai_api_key: Annotated[str, Header()]
) -> None:

    logger.info("Loading API key...")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    logger.info("Loading pdf files...")
    byte_files = [io.BytesIO(await f.read()) for f in files]

    logger.info("Parsing pdf files into string...")
    raw_text = get_pdf_text(byte_files)

    logger.info("Converting text data into chunks...")
    chunks = get_text_chunks(raw_text)

    # Create vector store
    global VECTORSTORE
    if VECTORSTORE is None:
        VECTORSTORE = get_vectorstore(chunks)
    else:
        VECTORSTORE.add_texts(chunks)


# Chat
@app.post("/chat")
async def chat(data: Data) -> None:

    logger.info("Checking for existing vector store...")
    if VECTORSTORE is None:
        raise HTTPException(
            status_code=400,
            detail="Please check your API key and upload your documents.",
        )

    logger.info("Checking for existing chain...")
    global CONV_CHAIN
    if CONV_CHAIN is None:
        CONV_CHAIN = get_conversation_chain(
            vectorstore=VECTORSTORE, temperature=data.temperature
        )

    logger.info("Performing QA...")
    answer = CONV_CHAIN({"question": data.question})
    chat_history = answer["chat_history"]
    chat_history = [message.content for message in chat_history]
    source_docs = answer["source_documents"]
    source_docs = [doc.page_content for doc in source_docs]
    json_payload = json.dumps(
        {"chat_history": chat_history, "source_documents": source_docs}
    )

    return Response(content=json_payload, media_type="application/json")


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8080, reload=True)
