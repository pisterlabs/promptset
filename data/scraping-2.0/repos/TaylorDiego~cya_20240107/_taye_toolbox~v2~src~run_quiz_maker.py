# import subprocess
import os
# import url_loader
# import question_maker
# from openai import OpenAI

def run_script(script_name):
    """Run the given script"""
    os.system(f"python {script_name}")

if __name__ == '__main__':
    run_script('url_loader.py')
    run_script('question_maker.py')
    run_script('upload_formatter.py')
