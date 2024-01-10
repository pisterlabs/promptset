from langchain.llms import OpenAI
import os
from langchain import PromptTemplate
import subprocess
import itertools
import threading
import time
import sys
import signal
import genfilename as gen
from userinput import getInput




openai = OpenAI(
    model_name="text-davinci-003",
    #model_name="gpt-4",
    temperature = 0,
)

#Catch Control C
def signal_handler(sig, frame):
    update_global(True)
    print('\nYou pressed Ctrl+C!')
    print('Graceful Exit\n')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Anamation FLAG
anamation_done = True 

def update_global(b:bool):
    global anamation_done
    anamation_done = b 
label = ""
def update_label(l:str):
    global label
    label =l


## Code Gen Prompt
template = """prompt = 
You are a genius python programmer code generater. Below message is divided into three Sections.
Below you will find three sections:
++ Example Input Code Section
This section will consist of example input those results in the output in the ++ Example Output Section
++ Example Output Section
The python code generated in this section is based on the information in the Example Input Code Section
++ Input GenerationCode Section
Input to generate code.
Do not improve or make any changes to the python code.

Start Here

++ Example Input Code Section
name = get_best_performing
description = "Useful for when you need to the performance of multiple stocks over a period. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over"

def get_best_performing(stocks, days_ago):
    best_stock = None
    best_performance = None
    for stock in stocks:
        try:
            performance = calculate_performance(stock, days_ago)
            if best_performance is None or performance > best_performance:
                best_stock = stock
                best_performance = performance
        except Exception as e:
            print("Could not calculate performance for ",stock)
++ Example Output Section
class StockBestPerformingInput(BaseModel):
    #Input for Stock ticker check. for percentage check

    stocktickers: List[str] = Field(..., description="Ticker symbols for stocks or indices")
    days_ago: int = Field(..., description="Int number of days to look back")

class StockGetBestPerformingTool(BaseTool):
    name = "get_best_performing"
    description = "Useful for when you need to the performance of multiple stocks over a period. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stocktickers: List[str], days_ago: int):
        price_change_response = get_best_performing(stocktickers, days_ago)

        return price_change_response

    def _arun(self, stockticker: List[str], days_ago: int):
        raise NotImplementedError("This tool does not support async")
    args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput
++ Input GenerationCode Section
Question: {query}

Answer: """
prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

def blankScreen():
    os.system('cls' if os.name == 'nt' else 'clear')

def openAI(prompt_template, user_input):
      return  openai(
          prompt_template.format(
          query=user_input
           )
        )
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if anamation_done:
            break
        sys.stdout.write('\r' + label + c)
        sys.stdout.flush()
        time.sleep(0.1)

def write_to_file(filename, content):
    with open(filename, "w") as file:
        file.write(content)

def generate_code():
    blankScreen()
    while True:
        update_global(True)
        user_input = getInput()

        if len(user_input) < 12:
            print("Invalid Input")
            break
        if user_input == "":
            break

        if user_input == ".clear":
            blankScreen()
            continue

        
        update_global(False)
        update_label("Generating Code ")
        t = threading.Thread(target=animate)
        t.start()

        result =  openAI(prompt_template, user_input)
        update_global(True)
        time.sleep(0.25) 

        print("\n----------------Generated Code --------")
        print(result, "\n\n")

        update_global(False)
        update_label("Generating File name: ")
        t = threading.Thread(target=animate)
        t.start()

        gname = gen.genfilename(user_input)
        done = True
        time.sleep(0.25)

        fname = f"{gname}.py"
        print("\nWriting Generated code to file: ", fname)

        write_to_file(fname, result)
        print()

generate_code()
print("END OF EXECUTION")
quit()
