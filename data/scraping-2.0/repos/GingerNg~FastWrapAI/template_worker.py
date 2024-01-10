import uvicorn
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from typing import List, Optional, Dict
from dataclasses import asdict, dataclass
from typing import List

from sse_starlette.sse import EventSourceResponse
from utils.env_utils import EnvKeys, EnvContext, app
from utils.openai_utils import Openai, num_tokens_from_messages
from utils.logger_utils import get_logger
from common.protocol.worker_api_protocol import WorkerGeneratorPath, CommonVo, EmbeddingRet, CompletionRet, StreamCompletionRet, UsageInfo
from common.protocol.openai_api_protocol import OpenaiGeneratorPath
from common.protocol.worker_api_protocol import svc_rd
import sys
from common.protocol.worker_api_protocol import release_model_semaphore, acquire_model_semaphore, create_background_tasks

logger = get_logger()
import os
os.environ['TRANSFORMERS_CACHE']='/***/***/cache/huggingface/'
os.environ['SENTENCE_TRANSFORMERS_HOME']='/***/***/cache/huggingface/'
import numpy as np
from common.protocol.worker_api_protocol import GenBase
from common.factory import get_obj



@app.post(WorkerGeneratorPath.TEXT_COMPLETION_STREAM.value)
async def api_generate_completion_stream(request: Request):
    try:
        params = await request.json()
        await acquire_model_semaphore()
        generator = generator_obj.generate_completion_stream(params)
        background_tasks = create_background_tasks()
        return StreamingResponse(generator, background=background_tasks)
    except Exception as e:
        logger.error(e)
        release_model_semaphore()
        raise HTTPException(status_code=500, detail=e.__repr__())

@app.post(WorkerGeneratorPath.TEXT_COMPLETION.value)
async def api_generate_completion(request: Request):
    try:
        params = await request.json()
        await acquire_model_semaphore()
        completion = generator_obj.generate_completion(params)
        background_tasks = create_background_tasks()
        return JSONResponse(content=completion.dict(), background=background_tasks)
    except Exception as e:
        logger.error(e)
        release_model_semaphore()
        raise HTTPException(status_code=500, detail=e.__repr__())

@app.post(WorkerGeneratorPath.TEXT_EMBEDDING.value)
async def api_generate_embedding(request: Request):
    try:
        params = await request.json()
        logger.debug(f"params: {params}")
        await acquire_model_semaphore()
        resp = generator_obj.generate_embedding(params=params)
        background_tasks = create_background_tasks()
        return JSONResponse(content=resp.dict(), background=background_tasks)
    except Exception as e:
        logger.error(e.__repr__())
        release_model_semaphore()
        raise HTTPException(status_code=500, detail=e.__repr__())

import sys
def usage():
    """
    print usage message and exit.
    """
    print('Usage: {} generator_class_path svc-host svc-port'.format(sys.argv[0]))
    sys.exit(1)

import argparse
if __name__ == "__main__":
    if len(sys.argv) < 4:
        usage()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    logger.info(f"args: {args}")
    print(args.model)

    generator_class_path = sys.argv[1]
    generator_obj = get_obj(class_path=generator_class_path,
                                        kwargs=vars(args))
    host=sys.argv[2]
    svc_port = int(sys.argv[3])
    name = generator_obj.svc_name
    svc_rd.register_svc(name, f"{name}-{host}-{svc_port}",host, svc_port)
    uvicorn.run(app, host="0.0.0.0", port=svc_port)