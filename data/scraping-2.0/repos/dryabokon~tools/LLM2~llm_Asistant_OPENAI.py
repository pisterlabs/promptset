import pandas as pd
import time
import os
import yaml
import openai
from openai import OpenAI
from halo import Halo
#----------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_Logger
import tools_time_profiler
#----------------------------------------------------------------------------------------------------------------------
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
#from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
#----------------------------------------------------------------------------------------------------------------------

class Assistant_OPENAILLM(object):
    def __init__(self, filename_config,folder_out):
        with open(filename_config, 'r') as config_file:
            config = yaml.safe_load(config_file)
            openai.api_key = config['openai']['key']
            os.environ["OPENAI_API_KEY"] = config['openai']['key']

        self.engine = 'gpt-4-1106-preview' #"gpt-4-1106-preview","gpt-3.5-turbo"
        #self.engine = "gpt-3.5-turbo"
        self.client = OpenAI()
        self.LLM = ChatOpenAI(temperature=0, openai_api_key=config['openai']['key'], model_name=self.engine)

        self.client = OpenAI(organization='org-3RPa8REGk1GuWZb27bEEW2jh')

        # stream = self.client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": "Say this is a test"}],
        #     stream=True,
        # )
        # for chunk in stream:
        #     if chunk.choices[0].delta.content is not None:
        #         print(chunk.choices[0].delta.content, end="")


        self.L = tools_Logger.Logger(folder_out+'Assistant_OPENAILLM.csv')
        self.TP = tools_time_profiler.Time_Profiler(verbose=False)
        self.thread = None
        return
#----------------------------------------------------------------------------------------------------------------------
    def pretify_string(self,text,N=120):
        lines = []
        line = ""
        for word in text.split():
            if len(line + word) + 1 <= N:
                if line:
                    line += " "
                line += word
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        result = '\n'.join(lines)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def Q(self, prompt, temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0):

        messages = [{"role": "system","content": prompt}]
        #response = self.client.chat.completions.create(model=self.engine, messages=messages,temperature=temp, max_tokens=tokens, top_p=top_p, frequency_penalty=freq_pen,presence_penalty=pres_pen)
        #response = self.client.completions.create(model=self.engine, prompt=prompt)
        #res = response.choices[0].message.content

        stream = self.client.chat.completions.create(model=self.engine,messages=[{"role": "user", "content": prompt}],stream=True)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")

        res = ''

        return res
    # ----------------------------------------------------------------------------------------------------------------------
    def wait_for_run_completion(self,thread_id, run_id, sleep_interval=5):
        while True:
            try:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
                if run.completed_at:
                    break
            except Exception:
                break
            time.sleep(sleep_interval)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def create_assistant_book_reader(self,file_id):
        assistant = self.client.beta.assistants.create(name="Book Reader",instructions="You are an Educator. Your role is to evaluate student's knowledge of the book.",model=self.engine,tools=[{"type": "retrieval"}],file_ids=[file_id])
        return assistant.id
# ----------------------------------------------------------------------------------------------------------------------
    def create_assistant_code_analyzer(self,file_ids):
        if not isinstance(file_ids,list):
            file_ids = [file_ids]
        assistant = self.client.beta.assistants.create(name="Code Analyzer",instructions="You are SW Developer. Your role is to analyze the codebase and interpret the code.",model=self.engine,tools=[{"type": "code_interpreter"}, {"type": "retrieval"}],file_ids=file_ids)
        return assistant.id
# ----------------------------------------------------------------------------------------------------------------------
    def get_assisatans(self,verbose=True):
        res = self.client.beta.assistants.list()
        ass_data = [[r.name,r.instructions,r.id,r.created_at] for r in res]

        if verbose:
            df = pd.DataFrame(ass_data,columns=['name','instructions','id','created_at'])
            print(tools_DF.prettify(df,showindex=False))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def delete_assistant(self,assistant_id):
        self.client.beta.assistants.delete(assistant_id=assistant_id)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_runs(self,thread_id):
        runs = self.client.beta.threads.runs.list(thread_id)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_stats(self):
        #'https://api.openai.com/v1/threads/thread_fivEcwzIWKGdREG3ak0DeVkj/runs/run_vZK7kY9mCrGkZTU2AeTn1obI'
        # url_run = f'https://api.openai.com/v1/threads/{thread.id}/runs/{run.id}'
        # headers = {"Authorization": f"Bearer {openai.api_key}","OpenAI-Beta":"assistants=v1","User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Content-Type": "application/json"}
        # response_run = requests.get(url_run, headers=headers)
        # url_thread = f'https://api.openai.com/v1/threads/{thread.id}'
        # response_thread = requests.get(url_thread, headers=headers)
        # #List run steps
        # url_steps = f'https://api.openai.com/v1/threads/{thread.id}/runs/{run.id}/steps'
        # response_steps = requests.get(url_steps, headers=headers)
        # url_step = f'https://api.openai.com/v1/threads/{thread.id}/runs/{run.id}/steps/step_Gd28OwDDZZGFVeuzFRJboeci'
        # response_step = requests.get(url_step, headers=headers)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def Q_assistant(self,query,assistant_id,file_id=None,verbose=True):

        if verbose:
            print(query)
            self.TP.tic('Q_assistant',verbose=False)
            spinner = Halo(spinner={'interval': 100,'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']})
            spinner.start()

        messages = [{"role": "user", "content": query}]
        if file_id is not None:
            messages[0]['file_ids'] = [file_id]

        thread = self.client.beta.threads.create(messages=messages)
        self.L.write(thread.id)

        run = self.client.beta.threads.runs.create(thread_id=thread.id,assistant_id=assistant_id)
        self.L.write(run.id)
        self.wait_for_run_completion(thread.id, run.id, sleep_interval=5)
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        texts = [m.content[0].text.value for m in messages.data]

        if verbose:
            spinner.succeed(self.TP.print_duration('Q_assistant',verbose=False))
            res = texts[0]
            print(res)
            spinner.stop()

        return texts[0]
    # ----------------------------------------------------------------------------------------------------------------------