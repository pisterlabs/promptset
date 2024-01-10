#%%
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache, SQLiteCache

import time
import os
from dotenv import load_dotenv
load_dotenv('../.env')

set_llm_cache(SQLiteCache("cache.db"))
# %%

chat = ChatOpenAI(
    temperature=0.1
)

#%%

start_tick = time.time()
answer = chat.predict("떡볶이는 레시피 알려주세요")

print(answer)

print(f'elapsed time: {time.time() - start_tick}')

# %%
