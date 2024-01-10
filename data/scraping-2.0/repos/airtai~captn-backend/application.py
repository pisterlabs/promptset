from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

import captn.captn_agents  # noqa
import google_ads  # noqa

import openai_agent  # noqa

app = FastAPI()
app.include_router(openai_agent.router, prefix="/openai", tags=["OpenAI"])
app.include_router(google_ads.router, tags=["Google Ads"])
app.include_router(captn.captn_agents.router, tags=["Captn Agents"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)  # nosec [B104]
