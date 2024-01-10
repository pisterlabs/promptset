"""Main entrypoint for the kitchenGPT."""
from contextlib import asynccontextmanager

import click
import uvicorn
from fastapi import FastAPI
from openai import OpenAI

import kitchengpt
from kitchengpt.api.router import api_router
from kitchengpt.config import settings

tags_metadata = [
    {
        "name": "Chat",
        "description": "Natural language chat assistant.",
    },
    {
        "name": "Fridge",
        "description": "Frige operations.",
    },
    {
        "name": "Oven",
        "description": "Oven operations.",
    },
]


@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Context manager to handle the lifespan of the app."""
    app_.state.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    yield


app = FastAPI(
    title="KitchenGPT REST API",
    version="0.1.0",
    summary="Deadpond's favorite app. Nuff said.",
    description="""ChimichangApp API helps you do awesome stuff. 🚀

                ## Items

                You can **read items**.

                ## Users

                You will be able to:

                * **Create users** (_not implemented_).
                * **Read users** (_not implemented_).

                """,
    openapi_tags=tags_metadata,
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    lifespan=lifespan,
)

app.include_router(api_router)


@click.command()
@click.version_option(
    version=kitchengpt.__version__,
    prog_name="kitchenGPT",
    message="%(prog)s %(version)s",
)
def main():
    """Console script for kitchengpt."""
    uvicorn.run(
        "kitchengpt.__main__:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[
            "kictchengpt",
        ],
        reload_includes=[
            "*.py",
        ],
    )


if __name__ == "__main__":
    main()
