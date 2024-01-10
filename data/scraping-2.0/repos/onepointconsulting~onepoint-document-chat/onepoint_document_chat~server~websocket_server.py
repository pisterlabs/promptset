import socketio
import datetime

from pathlib import Path
from typing import List

from asyncer import asyncify
from aiohttp import web


from langchain.schema import Document

from onepoint_document_chat.config import cfg
from onepoint_document_chat.log_init import logger
from onepoint_document_chat.server.session import add_message, delete_session
from onepoint_document_chat.service.qa_service import answer_question, ResponseText
from onepoint_document_chat.service.embedding_generation import (
    load_pdfs,
    add_embeddings,
)
from onepoint_document_chat.service.qa_service import vst


CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
}
SUPPORTED_CONTENT_TYPES = ["application/pdf"]


sio = socketio.AsyncServer(cors_allowed_origins=cfg.websocket_cors_allowed_origins)
app = web.Application()
sio.attach(app)


routes = web.RouteTableDef()


@routes.get("/")
async def get_handler(_):
    raise web.HTTPFound("/index.html")


@routes.get("/upload")
async def get_handler(_):
    html = (cfg.ui_folder / "index.html").read_text()
    return web.Response(text=html, content_type="text/html")


@routes.get("/upload/files")
async def get_handler(_):
    response = []
    for f in cfg.data_folder.iterdir():
        mtime = datetime.datetime.fromtimestamp(
            f.stat().st_mtime, tz=datetime.timezone.utc
        )
        response.append(
            {
                "date": mtime.isoformat(),
                "name": f.name,
                "relative_url": f"/files/{f.name}",
            }
        )
    return web.json_response(response, headers=CORS_HEADERS)


@sio.event
def connect(sid, environ):
    logger.info("connect %s ", sid)


@sio.event
async def question(sid, data):
    logger.info("question %s: %s", sid, data)
    session = add_message(sid, data)
    try:
        res: ResponseText = await asyncify(answer_question)(
            data, session.messages_history_str()
        )
    except:
        logger.exception("Failed to get message from server")
        await sio.emit(
            "response",
            ResponseText(response="Sorry. Failed to process", sources="").json(),
            room=sid,
        )
        return
    logger.info("response: %s", res.response)
    await sio.emit("response", res.json(), room=sid)


@sio.event
def disconnect(sid):
    logger.info("disconnect %s", sid)
    delete_session(sid)


def send_error(description: str, status: int = 200) -> str:
    return web.json_response(
        {"status": "error", "description": description},
        status=status,
        headers=CORS_HEADERS,
    )


@routes.post("/upload")
async def upload_file(request):
    """
    Used to upload a single file. Expects the request to have a parameteer "file"
    """
    data = await request.post()
    file = data.get("file")
    if file is None:
        return send_error("Parameter 'file' missing")

    token = data.get("token")
    if token != cfg.webserver_upload_token:
        return send_error("Security token is wrong.", 403)

    file_name = file.filename
    file_content_type = file.content_type
    if file_content_type not in SUPPORTED_CONTENT_TYPES:
        return web.json_response(
            {
                "status": "error",
                "description": "Unsupported file tipe: {file_content_type}. These are the supported file types: {SUPPORTED_CONTENT_TYPES}",
            }
        )

    content = file.file.read()
    target_file: Path = cfg.webserver_upload_folder / file_name
    target_file.write_bytes(content)
    documents: List[Document] = await asyncify(load_pdfs)(cfg.webserver_upload_folder)
    await asyncify(add_embeddings)(documents, vst)
    (cfg.data_folder / file_name).write_bytes(content)
    target_file.unlink(missing_ok=True)
    return web.json_response(
        {"status": "ok", "loaded": len(documents)},
        headers=CORS_HEADERS,
    )


if __name__ == "__main__":
    app.add_routes(routes)
    app.router.add_static("/files/", path=cfg.data_folder.as_posix(), name="files")
    app.router.add_static("/", path=cfg.ui_folder.as_posix(), name="ui")
    web.run_app(app, host=cfg.websocket_server, port=cfg.websocket_port)
