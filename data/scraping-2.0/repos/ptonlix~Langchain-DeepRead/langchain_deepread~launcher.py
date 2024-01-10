"""FastAPI app creation, logger configuration and main API routes."""
import logging
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from injector import Injector

from langchain_deepread.paths import docs_path
from langchain_deepread.settings.settings import Settings
from langchain_deepread.server.summary.summary_router import summary_router
from langchain_deepread.server.crawler.crawler_router import crawler_router
from langchain_deepread.server.qa.qa_router import qa_router
from langchain_deepread.server.health.health_router import health_router
from langchain_deepread.server.moderation.moderation_router import moderation_router

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
logger = logging.getLogger(__name__)
# handler = TimedRotatingFileHandler(
#     filename=PROJECT_ROOT_PATH / "log/app.log",
#     when="midnight",
#     interval=1,
#     backupCount=7,
# )
# handler.suffix = "%Y-%m-%d.log"
# handler.encoding = "utf-8"
# logger.addHandler(handler)


def create_app(root_injector: Injector) -> FastAPI:
    # Start the API
    with open(docs_path / "description.md") as description_file:
        description = description_file.read()

        tags_metadata = [
            {
                "name": "Crawler",
                "description": "Support mainstream online article information crawler",
            },
            {
                "name": "Summary",
                "description": "Article content summary, support one sentence summary, article core points, article guide, article labels and other information",
            },
            {
                "name": "QA",
                "description": "Based on the article content online Q&A QA, recall the article content by question similarity and answer the question",
            },
            {
                "name": "Moderation",
                "description": "Check whether the user uses the content in accordance with the policy",
            },
            {
                "name": "Health",
                "description": "Simple health API to make sure the server is up and running.",
            },
        ]

        async def bind_injector_to_request(request: Request) -> None:
            request.state.injector = root_injector

        app = FastAPI(dependencies=[Depends(bind_injector_to_request)])

        def custom_openapi() -> dict[str, Any]:
            if app.openapi_schema:
                return app.openapi_schema
            openapi_schema = get_openapi(
                title="Langchain-DeepRead",
                description=description,
                version="0.1.0",
                summary="DeepRead is an intelligent summary platform that allows users to start intelligent reading like marking a plaque",
                contact={
                    "url": "https://github.com/ptonlix/Langchain-DeepRead",
                },
                license_info={
                    "name": "Apache 2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
                },
                routes=app.routes,
                tags=tags_metadata,
            )
            openapi_schema["info"]["x-logo"] = {
                "url": "https://lh3.googleusercontent.com/drive-viewer"
                "/AK7aPaD_iNlMoTquOBsw4boh4tIYxyEuhz6EtEs8nzq3yNkNAK00xGj"
                "E1KUCmPJSk3TYOjcs6tReG6w_cLu1S7L_gPgT9z52iw=s2560"
            }

            app.openapi_schema = openapi_schema
            return app.openapi_schema

        app.openapi = custom_openapi  # type: ignore[method-assign]

        app.include_router(summary_router)
        app.include_router(crawler_router)
        app.include_router(qa_router)
        app.include_router(health_router)
        app.include_router(moderation_router)

        settings = root_injector.get(Settings)
        if settings.server.cors.enabled:
            logger.debug("Setting up CORS middleware")
            app.add_middleware(
                CORSMiddleware,
                allow_credentials=settings.server.cors.allow_credentials,
                allow_origins=settings.server.cors.allow_origins,
                allow_origin_regex=settings.server.cors.allow_origin_regex,
                allow_methods=settings.server.cors.allow_methods,
                allow_headers=settings.server.cors.allow_headers,
            )

        return app
