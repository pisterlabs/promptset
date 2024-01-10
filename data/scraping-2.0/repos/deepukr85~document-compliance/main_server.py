import asyncio
import os
from typing import AsyncIterable, Awaitable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel

load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = ""
else:
    openai_api_key = os.environ.get("OPENAI_API_KEY")

app = FastAPI()

template = """
you are a document compliance checking agent. you will be given complaince regulations and a target document. You have to bold the terms
which comes under the terms to avoid category using <b></b> html tag and italicize the terms which comes under the recommended terms category using <i></i> html tag.


Below is the compliance regulation

Terms to avoid:
{terms_to_avoid}

Recommended Terms:
{recommended_terms}


Below is the input document:
{input_data}

Your Response:"""

terms_to_avoid = """Avoid the terms in this list for any marketing programs you create because only financial institutions licensed as banks can use them.
Stripe or [Your Brand] name, Bank account, Bank balance, Banking, Banking account, Banking product, Banking platform, Deposits, Mobile banking, [Your Brand] pays interest,
[Your Brand] sets interest rates, [Your Brand] advances funds,
Phrases that suggest your users receive banking products or services directly from bank partners, 
for example:Create a [Bank Partner] bank account, A better way to bank with [Bank Partner], Mobile banking with [Bank Partner]"""

recommended_terms = """You need to brand and communicate the nature of the product while being mindful of regulations. Refer to the following list of recommended terms to use in your messaging when building out your implementation of the product.
Money management, or money management account or solution, Cash management, or cash management account or solution, [Your brand] account,
Financial services, Financial account, Financial product, Financial service product, Store of funds, Wallet or open loop wallet, Stored-value account,
Open-Loop stored-value account, Prepaid access account, Eligible for FDIC “pass-through” insurance, Funds held at [Partner Bank], Member FDIC """

prompt_template = PromptTemplate.from_template(template)


async def send_message(input: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        streaming=True,
        api_key=openai_api_key,
        verbose=True,
        callbacks=[callback],
    )

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            print(f"Caught exception: {e}")
        finally:
            event.set()

    task = asyncio.create_task(wrap_done(
        model.agenerate(messages=[[HumanMessage(content=prompt_template.format(terms_to_avoid=terms_to_avoid, recommended_terms=recommended_terms, input_data=input))]]),
        callback.done),
    )

    async for token in callback.aiter():
        yield f"data: {token}\n\n"

    await task


class StreamRequest(BaseModel):
    """Request body for streaming."""
    input: str


@app.post("/stream")
def stream(body: StreamRequest):
    return StreamingResponse(send_message(body.input), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8000, app=app)