### OpenAIAssistant Utils
#%%
import os,time
from openai import OpenAI
import pandas as pd
#%%

class OpenAI_File_Manager(object):
    """
    OpenAI file manager class to upload; find ; delete files 

    """
    def __init__(self, client=None, api_key=None):
        if not api_key:
            api_key = os.environ['OPENAI_API_KEY']
        if not client:
            client = OpenAI(api_key=api_key)
        self.client = client
        
    def upload_file(self,up_file_path,purpose='assistants'):
        """upload a document to openai and return doc id and name"""
        assert os.path.exists(up_file_path)
        file = self.client.files.create(
            file=open(up_file_path,'rb'),
            purpose=purpose
        )
        return(file.id,file.filename)
    
    def _get_file_list(self):
        ''' retrieve file list '''
        file_list = self.client.files.list().data
        return file_list 
    
    def _get_file_info(self):
        files_data = self._get_file_list()
        files_data_dict = [i.dict() for i in files_data]
        files_info = pd.DataFrame(files_data_dict)
        
        return files_info
    
    def get_files_by_ids(self,file_ids):
        if not isinstance(file_ids,list):
            file_ids=[file_ids]
        res = []
        for f_i in file_ids:
            f_r = self.client.files.retrieve(f_i)
            res.append(f_r)
        
        return res
    
    def delete_files_by_ids(self,file_ids):
        if not isinstance(file_ids,list):
            file_ids=[file_ids]
            
        files_info_df = self._get_file_info()
        db_file_ids = files_info_df['id'].values
        res = []
        for f_i in file_ids:
            if f_i in db_file_ids:
                self.client.files.delete(f_i)
                res.append(f_i)
            else:
                print("{} does not exist on openai server, please double check.".format(f_i))
            
        if len(res)>0:
            print("{} has been removed from file server.".format(res))
            
    def get_files_info_by(self,filter_criteria={},return_fields=['id','filename'],to_dict=True,to_single_list=True):
        """
        Filter file info based on a dictionary of criteria.
        Parameters:
        filter_criteria (dict): A dictionary where keys are column names and values are lists of column values to filter by.
        Returns:
        pd.DataFrame or dict: Filtered DataFrame.
        """
        files_info_df = self._get_file_info()
        
        for column, values in filter_criteria.items():
            if column in files_info_df.columns:
                files_info_df = files_info_df[files_info_df[column].isin(values)]
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if return_fields:
            files_info_df = files_info_df[return_fields]
        
        if to_dict:
            files_info_df = files_info_df.to_dict(orient='records')
        
        if len(return_fields)==1 and to_dict and to_single_list:
            files_info_df = [i.get(return_fields[0]) for i in files_info_df]
            
        return files_info_df
    
    
    
    
class OpenAIAssistant_Base():
    def __init__(self, client=None,api_key=None):
        if not api_key:
            api_key = os.environ['OPENAI_API_KEY']
        if not client:
            client = OpenAI(api_key=api_key)
        self.client = client
        self.assistant = None
        
        self.FileManager = OpenAI_File_Manager(client=client,api_key=api_key)
    
    def _set_active_assistant(self,current_assistant):
        self.assistant = current_assistant
        print('set {} as current active assistant.'.format(current_assistant.name,))
    
    def create_assistant(self,name,description,model="gpt-4-1106-preview",tools=[{"type":"retrieval"}],set_to_current=True,**kwargs):
        new_assistant = self.client.beta.assistants.create(
                        #instructions="You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
                        name=name,
                        description=description,
                        tools=tools,
                        model=model,
                        **kwargs
                        )
        
        if set_to_current:
            self._set_active_assistant(new_assistant)
            print('New assistant created and set to current')
        return new_assistant
    
    def update_current_assistant(self,**kwargs):
        if self.assistant:
            self.assistant = self.client.beta.assistants.update(
                self.assistant.id,
                **kwargs
                # instructions="You are an HR bot, and you have access to files to answer employee questions about company policies. Always response with info from either of the files.",
                # name="HR Helper",
                # tools=[{"type": "retrieval"}],
                # model="gpt-4",
                # file_ids=["file-abc123", "file-abc456"],
                )
        else:
            raise('Current Assistant not set, please use _set_active_assistant to set current assistant.')
            
    def delete_assistants_by_ids(self,as_ids):
        if not isinstance(as_ids,list):
            as_ids=[as_ids]
            
        as_info_df = self._get_assistant_info()
        db_as_ids = as_info_df['id'].values
        res = []
        for a_i in as_ids:
            if a_i in db_as_ids:
                self.client.beta.assistants.delete(a_i)
                res.append(a_i)
            else:
                print("{} does not exist on openai server, please double check.".format(a_i))
            
        if len(res)>0:
            print("{} has been removed from file server.".format(res))
        
    def _get_assistant_list(self):
        ''' retrieve assistant list '''
        a_list = self.client.beta.assistants.list(
            order="desc",
            #limit="20"
        )
        
        return a_list
    
    def _get_assistant_info(self):
        """Get assistants meta info"""
        a_data = self._get_assistant_list()
        a_data_dict = [i.dict() for i in a_data]
        a_info = pd.DataFrame(a_data_dict)
        
        return a_info
    
    def get_assistants_by_ids(self,a_ids):
        if not isinstance(a_ids,list):
            a_ids=[a_ids]
        res = []
        for a_i in a_ids:
            a_r = self.client.beta.assistants.retrieve(a_i)
            res.append(a_r)
        if len(res) == 1:
            res = res[0]
        return res
    
    def get_assistants_info_by(self,filter_criteria={},return_fields=['id','name'],to_dict=True,to_single_list=True):
        """
        Filter file info based on a dictionary of criteria.
        Parameters:
        filter_criteria (dict): A dictionary where keys are column names and values are lists of column values to filter by.
        Returns:
        pd.DataFrame or dict: Filtered DataFrame. or a list 
        """
        as_info_df = self._get_assistant_info()
        
        for column, values in filter_criteria.items():
            if column in as_info_df.columns:
                as_info_df = as_info_df[as_info_df[column].isin(values)]
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if return_fields:
            as_info_df = as_info_df[return_fields]
        
        if to_dict:
            as_info_df = as_info_df.to_dict(orient='records')
        
        if len(return_fields)==1 and to_dict and to_single_list:
            as_info_df = [i.get(return_fields[0]) for i in as_info_df]
            
        return as_info_df
    
    def quick_run(self,user_input_dict,**kwargs):
        if self.assistant:
            run = self.client.beta.threads.create_and_run(
                        assistant_id=self.assistant.id,
                        thread={
                            "messages": [
                                            user_input_dict
                                            #{"role": "user", "content": "Explain deep learning to a 5 year old."} ## not sure if you can add files here 
                                        ]
                        },
                        **kwargs
                        #instructions= 'update system instruction on the fly '
                    )
        else:
            raise('no active assistant set, please use _set_activate_assistant to activate an assistant.')
        
        return run
    
    def _get_finished_run(self,initial_run):
        start_time = time.time()
        while True:
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(thread_id=initial_run.thread_id, run_id=initial_run.id)
            elapsed_time = time.time() - start_time
            print(f"\rElapsed time: {elapsed_time:.2f} s || Status: {run.status}    ", end="", flush=True)
            if run.status in ['completed', 'failed', 'requires_action']:
                return run
            
    def _parse_return_message(self,run):
        ## retrieve the message 
        return_messages = self.client.beta.threads.messages.list(
            thread_id=run.thread_id
        )
        # for each in return_messages:
        #   print(each.role+": {}".format(each.content[0].text.value))
        #   print("=============")
        return return_messages.data[0].content[0].text.value

    def quick_query(self,user_input_dict,**kwargs):
        """
        A quick query from scratch, no conversation history is used 
        
        Args:
            user_input_dict (_type_): a dictionary with system message and user message

        Returns:
            json string : a json string of responses 
        """
        init_run = self.quick_run(user_input_dict,**kwargs)
        post_run = self._get_finished_run(init_run)
        res = self._parse_return_message(post_run)
        
        return res,post_run.status
    
    def quick_query_and_parse(self,**kwargs):
        
        raise NotImplementedError("This function is a placeholder and needs to be implemented.")
    
    
#%%
    