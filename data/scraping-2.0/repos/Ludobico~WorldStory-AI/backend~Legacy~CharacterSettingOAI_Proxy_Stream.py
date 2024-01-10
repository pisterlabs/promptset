import configparser
import asyncio
from typing import AsyncIterable, Optional, List, Mapping, Any

from langchain.callbacks import AsyncIteratorCallbackHandler
# from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from Module.Template.BaseTemplate import base_template, few_shot_base_template
from langchain.schema import HumanMessage

import g4f
from g4f import Provider, models
from langchain.llms.base import LLM
from Legacy.G4FLLM import G4FLLM


async def send_message_OAI(content: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    BaseTemplateResult = base_template()
    FewShotTemplateResult = few_shot_base_template()

    prompt = PromptTemplate(
        template=BaseTemplateResult['template']+FewShotTemplateResult, input_variables=["instruct"])

    llm: LLM = G4FLLM(model=models.gpt_35_turbo,  provider=Provider.GptGo, verbose=True)
    model = LLMChain(prompt=prompt, llm=llm, callbacks=[callback],verbose=True)

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
