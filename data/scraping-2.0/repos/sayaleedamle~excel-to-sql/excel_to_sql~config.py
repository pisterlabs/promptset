from pathlib import Path
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
import psycopg2


load_dotenv()


class Config:
    model_name = os.getenv("OPENAI_MODEL")
    llm_cache = os.getenv("LLM_CACHE") == "True"
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=model_name,
        temperature=0,
        request_timeout=os.getenv("REQUEST_TIMEOUT"),
        cache=llm_cache,
        streaming=True,
    )
    verbose_llm = os.getenv("VERBOSE_LLM") == "True"
    python_executor = Path(os.getenv("PYTHON_SCRIPT"))

    #Database connection
    dbname = os.getenv("DB_NAME")
    user = os.getenv("USER")
    password = os.getenv("PWD")
    host = os.getenv("HOST")  
    port = os.getenv("PORT")  
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    
    path_excel = Path(os.getenv("EXCEL_PATH_DISC"))
    
    if not path_excel.exists():
        path_excel.mkdir(exist_ok=True, parents=True)

    ui_timeout = os.getenv("REQUEST_TIMEOUT")

cfg = Config()


if __name__ == "__main__":
    print("llm: ", cfg.llm)
