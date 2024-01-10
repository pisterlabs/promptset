import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional, Union

sys.path.append("/root/sherpa/exllamav2")
sys.path.append("/root/sherpa/guidance")

import elasticapm
import torch
import uvicorn
from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
from exllamav2_hf import Exllamav2HF
from exllama_hf import ExllamaHF
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

import guidance

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model path")
parser.add_argument("--port", type=int, required=True, help="api port")

# [init torch]:
torch.set_grad_enabled(False)
torch.cuda._lazy_init()
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_printoptions(precision=10)

# Setup FastAPI:
app = FastAPI()
# apm = make_apm_client(
#     {
#         "SERVER_URL": "https://apm.zuma.dev/",
#         "SERVICE_NAME": "sherpa",
#         "SECRET_TOKEN": "",
#         "SPAN_COMPRESSION_EXACT_MATCH_MAX_DURATION": "0ms",
#     }
# )
# app.add_middleware(ElasticAPM, client=apm)

semaphore = asyncio.Semaphore(1)


class GenerateRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def stream_data(req: GenerateRequest):
    with elasticapm.capture_span("acquire lock"):
        await semaphore.acquire()
    try:
        t0 = time.time()
        prompt = req.prompt
        with elasticapm.capture_span("guidance"):
            output = guidance(prompt)()
        t1 = time.time()
        _sec = t1 - t0
        print(f"Output generated in {_sec}")
        resp = output.variables()
        resp.pop("llm", None)
        resp.pop("@raw_prefix", None)
        resp.pop("logging", None)
        resp.pop("accelerate", None)
        return JSONResponse(resp)
    finally:
        semaphore.release()


# -------


if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.model
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    model = Exllamav2HF.from_pretrained(model_path)
    model.config = config

    guidance.llm = guidance.llms.Transformers(
        model,
        tokenizer,
        caching=False,
        acceleration=False,
    )
    guidance.llm.device = None  # Prevent from messing with GPU

    uvicorn.run(app, host="0.0.0.0", port=args.port)
