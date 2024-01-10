from pandasai import SmartDataframe
from pandasai.llm import OpenAI




def createPlotForStreamlit(CSV_FILE,prompt):
    llm = OpenAI()
    df = SmartDataframe(CSV_FILE, config={"llm": llm,"enable_cache": False,"save_logs":False,"open_charts":False})
    return  df.chat(prompt)
