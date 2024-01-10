from fastapi import FastAPI
from fastapi.logger import logger as fastapi_logger
import uvicorn
from logging.handlers import RotatingFileHandler
import logging
import os
from datetime import datetime
from LangchainAgent import run_agent
from typing import List
import json
from time import sleep
from fastapi.responses import JSONResponse

app = FastAPI()

DEBUG = False

@app.get("/query")
def query(q: str, context: str = "[]") -> dict:
    """Query the LangchainAgent with a question and context."""
    # dejsonify context
    context = json.loads(context)
    if DEBUG:
        summary = "test"
        # sleep(10)
        message = "Based on the available information, it does not appear that the debt of FRGI (Fiesta Restaurant Group) in 2016 is explicitly stated. The financial data from the selected consolidated financial statements for each of the years ending on January 3, 2016, and December 28, 2014, shows figures such as net cash provided from operating activities, net cash used for investing activities, net cash provided by (used for) financing activities, and total capital expenditures, but does not provide specific details about the debt. The balance sheet data also does not explicitly state the debt amount for 2016. Additional information or specific disclosures regarding the debt of FRGI in 2016 may be needed. [https://www.sec.gov/Archives/edgar/data/1534992/000153499216000043/frgi-20150103x10k.htm] [https://www.sec.gov/Archives/edgar/data/1534992/000153499216000043/frgi-20150103x10k.htm] [https://www.sec.gov/Archives/edgar/data/1534992/000153499216000043/frgi-20150103x10k.htm] [https://www.sec.gov/Archives/edgar/data/1534992/000153499216000043/frgi-20150103x10k.htm]"
        # message = "FRGI experienced a decline in its stock prices in 2016. The high and low closing prices of FRGI's common stock for 2016 were as follows:\n- First Quarter: High of $66.99 and Low of $55.32.\n- Second Quarter: High of $62.32 and Low of $46.45.\n- Third Quarter: High of $58.47 and Low of $46.35.\n- Fourth Quarter: High of $45.71 and Low of $32.01.\n\nFRGI did not pay any cash dividends in 2015 or 2014 and does not anticipate paying any cash dividends in the foreseeable future. [https://www.sec.gov/Archives/edgar/data/1534992/000153499216000043/frgi-20150103x10k.htm]\n\nSelected financial data for FRGI in 2016:\n- Net cash provided from operating activities: $81,352,000.\n- Net cash used for investing activities: ($87,671,000).\n- Net cash provided by (used for) financing activities: $6,513,000.\n- Total capital expenditures: ($87,570,000).\n- Total assets: $415,645,000.\n- Working capital: ($15,067,000). [https://www.sec.gov/Archives/edgar/data/1534992/000153499216000043/frgi-20150103x10k.htm]\n\nPlease note that historical results are not necessarily indicative of future results. [https://www.sec.gov/Archives/edgar/data/1534992/000153499216000043/frgi-20150103x10k.htm]"
        # message = "Restoration Hardware has multiple credit facilities including an asset based credit facility and term loan credit agreements. The ABL Credit Agreement was amended on July 29, 2021, providing a revolving line of credit with initial availability of up to $600 million and includes an accordion feature which may be expanded up to $900 million. The Credit Agreement established a $80 million last-in, last-out term loan facility. As of January 29, 2022, the asset based credit facility has deferred financing fees of $4.1 million and the term loan credit agreement has outstanding amounts of $2.0 billion. Additionally, the company has equipment promissory notes outstanding with payments due in fiscal 2023. Overall, Restoration Hardware's balance sheets show significant debt and credit facilities to manage their operations."
    else:
        message, summary = run_agent(q, context)
    return {"message": message, "summary": summary}


if __name__ == '__main__':
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s", "%Y-%m-%d %H:%M:%S")
    # create log file
    if not os.path.exists('./dblog'):
        os.makedirs('./dblog')
    # touch new log file based on current datetime
    log_file = datetime.now().strftime('./dblog/%Y-%m-%d_%H-%M-%S.log')
    handler = RotatingFileHandler(log_file, backupCount=0)
    logging.getLogger().setLevel(logging.NOTSET)
    fastapi_logger.addHandler(handler)
    handler.setFormatter(formatter)

    fastapi_logger.info('****************** Starting Server *****************')
    uvicorn.run(app, port=8081)
