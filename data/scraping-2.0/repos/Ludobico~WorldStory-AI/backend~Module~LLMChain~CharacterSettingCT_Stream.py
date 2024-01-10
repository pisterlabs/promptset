import asyncio
import os
from typing import AsyncIterable
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.ctransformers import CTransformers
from Module.Template.BaseTemplate import base_template


async def send_message(ct_params) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    BaseTemplateResult = base_template()

    prompt = PromptTemplate(
        template=BaseTemplateResult['template'], input_variables=["instruct"])

    # llm = LlamaCpp(
    #     model_path="./Models/WizardLM-13B-1.0.ggmlv3.q4_0.bin",
    #     callbacks=[callback],
    #     verbose=True,
    #     streaming=True,
    #     max_tokens=25,
    # )
    testconfig = {"top_k": ct_params.top_k, "top_p": ct_params.top_p, "temperature": ct_params.temperature,
                  "last_n_tokens": ct_params.last_n_tokens, "max_new_tokens": ct_params.max_new_tokens, "gpu_layers": ct_params.gpu_layers}
    llm = CTransformers(
        model=f"./Models/{ct_params.model_name}", model_type="llama", callbacks=[callback], verbose=True, config=testconfig)

    model = LLMChain(prompt=prompt, llm=llm, verbose=True)

    question = BaseTemplateResult['instruct']

    task = asyncio.create_task(
        model.arun(question)
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task
