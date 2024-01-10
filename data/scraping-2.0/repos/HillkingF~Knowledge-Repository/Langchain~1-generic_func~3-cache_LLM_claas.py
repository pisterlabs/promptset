'''
@Author: 冯文霓
@Date: 2023/6/7
@Purpose: 介绍如何缓存LLm调用，常用的方法，内存缓存、Redis、GPT缓存、momento、SQL数据库魂村
'''

from Langchain.units import *

import langchain
from langchain.llms import OpenAI
import time
# 1.内存缓存
from langchain.cache import InMemoryCache
# 2.sqlLite缓存
from langchain.cache import SQLiteCache
# 3.Redis缓存
from redis import Redis
from langchain.cache import RedisCache
# 4.GPTCache 缓存


# 记录开始时间
star_time = time.time()

# To make the caching really obvious, lets use a slower model.
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
langchain.llm_cache = InMemoryCache()


# The first time, it is not yet in cache, so it should take longer
print(llm("Tell me a joke"))

# 记录结束时间
end_time = time.time()
# 计算运行时间差值
eps_time = end_time - star_time

# 第二次运行会比第一次运行更快
print(eps_time)