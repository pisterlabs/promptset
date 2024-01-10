from application import Application
from sqlServerDB import SQLServer
import openai

def main():
    """_summary_
    """    
    #Questions
    # give me the product names that have product number as FR-R92B-58
    print("Input a question")
    app = Application()
    while True:
        input_console = input("Question:")
        try:
            output = app.workflow(message=input_console, sender="USER")
        except ValueError:
            print("Invalid input.")

if __name__ == "__main__":
    main()