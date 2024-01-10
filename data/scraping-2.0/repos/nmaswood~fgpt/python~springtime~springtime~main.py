import uvicorn
from anthropic import Anthropic
from fastapi import FastAPI
from loguru import logger

from springtime.excel.table_extractor import TabulaTableExtractor
from springtime.models.open_ai import OpenAIModel
from springtime.object_store.object_store import GCSObjectStore
from springtime.routers.chat_router import ChatRouter
from springtime.routers.embeddings_router import EmbeddingsRouter
from springtime.routers.pdf_router import PdfRouter
from springtime.routers.prompt_router import PromptRouter
from springtime.routers.report_router import ReportRouter
from springtime.routers.table_router import TableRouter
from springtime.routers.text_router import TextRouter
from springtime.routers.vector_router import VectorRouter
from springtime.services.chat_service import OpenAIChatService
from springtime.services.embeddings_service import OpenAIEmbeddingsService
from springtime.services.excel_analyzer import ClaudeExcelAnalyzer
from springtime.services.prompt_service import PromptServiceImpl
from springtime.services.report_service import OpenAIReportService
from springtime.services.scan_service import OpenAIScanService
from springtime.services.sheet_processor import (
    CLAUDE_SHEET_PROCESSOR,
)
from springtime.services.table_analyzer import TableAnalyzerImpl
from springtime.services.thumbnail_service import FitzThumbnailService
from springtime.services.vector_service import PineconeVectorService

from .settings import SETTINGS

app = FastAPI()


logger.info("Starting server")
if SETTINGS.tracing_enabled:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    from springtime.tracer import init_tracing

    init_tracing()
    FastAPIInstrumentor.instrument_app(app)


OBJECT_STORE = GCSObjectStore()
TABLE_EXTRACTOR = TabulaTableExtractor(OBJECT_STORE)

THUMBNAIL_SERVICE = FitzThumbnailService()
ANTHROPIC_CLIENT = Anthropic()

SCAN_SERVICE = OpenAIScanService(OpenAIModel.gpt3_16k)
CLAUDE_EXCEL_ANALYZER = ClaudeExcelAnalyzer(ANTHROPIC_CLIENT)

CLAUDE_TABLE_ANALYZER = TableAnalyzerImpl(CLAUDE_EXCEL_ANALYZER, CLAUDE_SHEET_PROCESSOR)


EMBEDDING_SERVICE = OpenAIEmbeddingsService()
VECTOR_SERVICE = PineconeVectorService(
    api_key=SETTINGS.pinecone_api_key,
    environment=SETTINGS.pinecone_env,
    index_name=SETTINGS.pinecone_index,
    namespace=SETTINGS.pinecone_namespace,
)
OPENAI_REPORT_SERVICE = OpenAIReportService()
CHAT_SERVICE = OpenAIChatService(OpenAIModel.gpt3_16k)

PROMPT_SERVICE = PromptServiceImpl(ANTHROPIC_CLIENT)


app.include_router(ChatRouter(CHAT_SERVICE).get_router())
app.include_router(
    ReportRouter(
        OPENAI_REPORT_SERVICE,
        SCAN_SERVICE,
    ).get_router(),
)
app.include_router(
    PdfRouter(TABLE_EXTRACTOR, OBJECT_STORE, THUMBNAIL_SERVICE).get_router(),
)
app.include_router(
    TableRouter(
        CLAUDE_TABLE_ANALYZER,
        OBJECT_STORE,
    ).get_router(),
)
app.include_router(TextRouter().get_router())
app.include_router(VectorRouter(VECTOR_SERVICE).get_router())
app.include_router(EmbeddingsRouter(EMBEDDING_SERVICE).get_router())


app.include_router(PromptRouter(PROMPT_SERVICE).get_router())


@app.get("/ping")
def ping():
    return {"ping": "ping"}


@app.get("/healthz")
def healthz():
    return "OK"


def start():
    uvicorn.run(
        "springtime.main:app",
        host=SETTINGS.host,
        port=SETTINGS.port,
        reload=SETTINGS.reload,
        timeout_keep_alive=60,
    )
