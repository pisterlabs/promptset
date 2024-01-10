from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool
tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
openai.api_key = ""
llm = OpenAIChat(model_name="gpt-3.5-turbo")
chain_new = APIChain.from_llm_and_api_docs(llm,open_meteo_docs.OPEN_METEO_DOCS,verbose=True)

#我们向ChatGPT询问上海当前的温度
chain_new.run('上海现在的温度是多少摄氏度？')