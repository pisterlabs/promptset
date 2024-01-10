import os
import openai
import LLMs as LLM
import VectorDB as M
from getpass import getpass
import CreateProject as CP
import traceback







def main():
    CP.FinTune_create_project()
#    gen_subfolders("Make python helloworld program")
 

  
def TestMemory():
   
   print( M.SaveVectorDB("ClassName","MethodName"))
    


if __name__ == "__main__":
    main()
