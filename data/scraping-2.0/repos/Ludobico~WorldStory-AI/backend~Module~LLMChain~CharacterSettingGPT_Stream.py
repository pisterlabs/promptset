import asyncio
from typing import AsyncIterable

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from Module.Template.BaseTemplate import base_template, few_shot_base_template
from Module.Template.FewShotPromptForCharSetting import character_setting_examples
from Module.CharacterCheck import CharacterConfig
from Module.LLMChain.CustomLLM import CustomLLM_GPT, CustomLLM_Llama, CustomLLM_FreeGPT

async def character_setting_gpt_stream(content : str) -> AsyncIterable[str]:
  callback = AsyncIteratorCallbackHandler()
  llm = CustomLLM_FreeGPT()
  BaseTemplateResult = base_template()
  FewShotTemplateResult = few_shot_base_template()
  user_preference = CharacterConfig.user_config_parser()

  # prompt = PromptTemplate(template=BaseTemplateResult['template'] + FewShotTemplateResult, input_variables=["instruct"])
  prompt = PromptTemplate(template=BaseTemplateResult['template'], input_variables=["instruct", "name", "gender", "era"])

  chain = LLMChain(llm=llm, prompt=prompt)
  question = BaseTemplateResult['instruct']
  name = user_preference['name']
  gender = user_preference['gender']
  era = user_preference['era']

  task = asyncio.create_task(chain.arun(instruct=question,name=name,gender=gender,era=era ,callbacks=[callback]))
  
  try:
      async for token in callback.aiter():
          yield token
  except Exception as e:
      print(f"Caught exception: {e}")
  finally:
      callback.done.set()

  await task