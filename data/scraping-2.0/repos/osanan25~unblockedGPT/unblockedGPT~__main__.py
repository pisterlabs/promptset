import os
from unblockedGPT.typeresponse import Typeinator
from unblockedGPT.auth import Database
import openai
import time
import sys
from unblockedGPT.typeGPT import typeGPT
from unblockedGPT.saveResponse import heroSave, stealthSave
from unblockedGPT.saveKeysCmd import saveKeysCmd, saveKeys64Cmd
def run():
    cur_dir = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    app_path = os.path.join(dir_path, 'app.py')
    os.system(f'streamlit run {app_path} -- "{cur_dir}"')

def setKeys(args = sys.argv):
    if "-h" in args or "--help" in args:
        print("Usage: setkeys -b64(optional signals a base64 encoded key)")
        return
    if "-b64" in args:
        saveKeys64Cmd()
        return
    saveKeysCmd()

def stealthTypeCmd(args = sys.argv):
    if '-h' in args or '--help' in args:
        print("Usage: stealthtype -p [path to text file/file in curent dir] ")
        return
    if "-p" in args:
        stealthSave(args[args.index("-p") + 1], os.getcwd())
    else:
        print("No file path provided. Use -h for help")
        return


def heroTypeCmd(args = sys.argv):
    if "-h" in args or "--help" in args:
        print("Usage: herotype -p [path to text file/file in curent dir]")
        return
    if "-p" in args:
        path = args[args.index("-p") + 1]
        heroSave( path,os.getcwd())
    else:
        print("No file path provided. Use -h for help")
        return

def textToType(args = sys.argv):
    if "-h" in args or "--help" in args:
        print("Usage: typetext -p [path to text file/file in curent dir] -t [time in minutes] (optional)")
        return
    if "-p" in args:
        path = args[args.index("-p") + 1]
        if os.path.exists(path):
            with open(path, 'r') as file:
                text = file.read()
        else:
            curentDir = os.getcwd()
            #add path to current directory
            path = os.path.join(curentDir, path)
            if os.path.exists(path):
                with open(path, 'r') as file:
                    text = file.read()
            else:
                print("File path provided does not exist. Use -h for help")
                return
    else:
        print("No file path provided. Use -h for help")
        return
    if "-t" in args:
        timeInput = int(args[args.index("-t") + 1])
    else:
        timeInput = -1
    typer = Typeinator()
    print("Typing in 5 seconds...")
    time.sleep(5)
    if timeInput == -1:
        typer.type(text)
    else:
        typer.timeToType(text, timeInput)
    return
    
def typeGPTCmd():
    typeGPT()
if __name__ == '__main__':
    type()