import os
import io
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

load_dotenv()

API_KEY = os.getenv("API_KEY")

class UnrecognizedFileType(Exception):
    """Exception for an unrecognized file type"""
    pass


class AIAgent:
    def __init__(self):
        self.agent = None

    async def upload_files(self, file, sheet_name):
        df = await read_file(file, sheet_name)
        if df is not None:
            self.agent = create_ai_agent(df)
            return 'Files uploaded successfully and AI agent is ready!'
        else:
            return "No valid data file was found in the uploaded files."

    def run_agent(self, question):
        return self.agent.run(question)


async def check_file_type(file):
    file_ = file[0]
    filename = file_.filename
    print(f'file name:  {filename}')
    extension = filename.rsplit('.', 1)[-1].lower()
    print(f'extension:  {extension.upper()}')
    content = await file_.read()

    if extension in ['csv', 'json', 'xlsx', 'xls']:
        return  extension, content
    else:
        raise UnrecognizedFileType(f"The file '{filename}' is not a recognized data file."
                                   f" It has a {extension.upper()} extension.")


async def read_file(file, sheet_name):
    extension, content = await check_file_type(file)

    if extension == 'csv':
        return pd.read_csv(io.StringIO(content.decode()))
    elif extension == 'json':
        return pd.read_json(io.StringIO(content.decode()))
    elif extension in ['xlsx', 'xls']:
        return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)


def create_ai_agent(df):
    """Create AI agent with given dataframe"""
    chat_model = ChatOpenAI(openai_api_key=API_KEY,
                            model='gpt-3.5-turbo',
                            temperature=0.0)
    return create_pandas_dataframe_agent(chat_model, df, verbose=True)
