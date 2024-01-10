from loguru import logger

from langchain.callbacks.file import FileCallbackHandler
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

logfile = 'output.log'
logger.add(logfile, colorize=True, enqueue=True, )
handler = FileCallbackHandler(filename=logfile)

llm = OpenAI()
prompt = PromptTemplate.from_template(template='1+{number}=')

chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler, ], verbose=True)
answer = chain.run(number=2)
logger.info(answer)
