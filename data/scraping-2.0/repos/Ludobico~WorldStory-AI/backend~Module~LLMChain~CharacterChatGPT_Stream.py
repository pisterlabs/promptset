import configparser
import asyncio
from typing import AsyncIterable, Optional, List, Mapping, Any

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from Module.Template.BaseTemplate import chat_base_template
from Module.CharacterCheck import CharacterConfig
from Module.LLMChain.CustomLLM import CustomLLM_GPT, CustomLLM_Llama, CustomLLM_FreeGPT

async def chat_with_OAI(content: str, char_prompt_path, memory) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    chat_base_template_result = chat_base_template(char_prompt_path)
    user_config = CharacterConfig.user_config_parser()
    user_name = user_config['user_name']
    user_lang = user_config['user_lang']

    prompt = PromptTemplate(
        template=chat_base_template_result['chat_template'], input_variables=["char_prompt", "message", "chat_history", "user_name", "ai_name", "user_lang"])

    llm =  CustomLLM_FreeGPT()
    # memory = ConversationBufferMemory(memory_key="chat_history", input_key="message")
    model = LLMChain(prompt=prompt, llm=llm, memory=memory)

    char_prompt = chat_base_template_result['char_prompt']
    question = content
  
    task = asyncio.create_task(
        model.arun(char_prompt = char_prompt, message = question,
                    user_name = user_name, ai_name = char_prompt_path,
                      user_lang = user_lang, callbacks=[callback])
    )
    try:
        async for token in callback .aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task
