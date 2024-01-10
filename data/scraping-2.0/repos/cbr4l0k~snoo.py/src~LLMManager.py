import json
import os
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from FileHandler import FileHandler
from prompt_handler import PromptHandler

# some important enviroment variables
load_dotenv()
PROJECTS_PATH = os.getenv("PROJECTS_PATH")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OUTPUTS_PATH = os.getenv("OUTPUTS_PATH")
DEFAULT_LLM = os.getenv("DEFAULT_LLM")

import warnings
warnings.filterwarnings("ignore")


class LLM:

    """
        This class is a wrapper for the langchain library.
        It is used to generate explainations for code.
        It uses the langchain library to generate the explainations.

        It is also used to generate cohesion and coupling analysis for a project.
        It also produces explainations for the folders in the project.
        It is also used to generate explainations for a directory.
        It is also used to generate explainations for a file.


        Attributes:
            model: The langchain model used to generate the explainations.
            llm_chain: The langchain chain used to generate the explainations.
            context_window_size: The size of the context window used to generate the explainations.
            options: The options used to initialize the langchain model.
            prompt_handler: The prompt handler used to generate the prompts for the langchain model.
            file_handler: The file handler used to handle the files in the project.
        
        Methods:
            load_model: Loads the langchain model.
            load_chain: Loads the langchain chain.
            generate_response: Generates an explaination for a file.
            generate_explaination_for_directory: Generates an explaination for a directory.
            generate_cohesion_coupling_analysis: Generates a cohesion and coupling analysis for a project.
            set_context_window_size: Sets the context window size for the langchain model.
            _check_response: Checks the response of the langchain model.
    """

    projects_path = ''

    def __init__(self, projects_path: str, options: dict) -> None:
        self.model : OpenAI = None
        self.llm_chain : LLMChain = None
        self.context_window_size : int = None
        self.options : dict = options
        self.prompt_handler : PromptHandler = PromptHandler(model_name=self.options["model_name"])
        
        self.prompt_handler.set_projects_path(projects_path)
        self.set_projects_path(projects_path)

        self.load_model()
        self.file_handler : FileHandler = FileHandler()

    @staticmethod
    def set_projects_path(projects_path: str) -> None:
        """
            Sets the outputs path for the LLMManager.
        """
        LLM.projects_path = projects_path

    def set_context_window_size(self, context_window_size: int) -> None:
        """
            Sets the context window size for the langchain model.

            Args:
            ----------
            context_window_size: int
                The size of the context window to use.

            Returns:
            ----------
            None
        """
        self.context_window_size = context_window_size

    def load_model(self) -> None:
        """
            Loads the OpenAI langchain model.
            
            Args:
            ----------
            None

            Returns:
            ----------
            None
        """
        model = OpenAI(**self.options)
        self.prompt_handler.set_model(model_name=self.options["model_name"])
        self.model = model

    def load_chain(self, template: dict[str, any], requires_memory: bool = False) -> None:
        """
            Loads the langchain chain.

            Args:
            ----------
            template: dict[str, any]
                The template to use for the langchain chain.
            requires_memory: bool
                Whether the langchain chain requires memory or not.
            
            Returns:
            ----------
            None
            
        """
        
        self.load_model()
        prompt: PromptTemplate = PromptTemplate(
            input_variables=template["input_variables"],
            template=template["template"]
        )

        llm_chain: LLMChain = LLMChain(
            llm=self.model,
            prompt=prompt,
            verbose=self.options["verbose"]
        )

        self.llm_chain = llm_chain

    def generate_response(self, file_full_path: str, code: str) -> str:
        """
            Generates an explaination for a file.

            Args:
            ----------
            file_full_path: str
                The full path to the file.
            code: str
                The code to generate the explaination for.
            
            Returns:
            ----------
            str
                The explaination for the file.
        """

        # estimate the number of tokens for the code
        code_token_size = (self.context_window_size - self.prompt_handler.longest_prompt_lenght)

        # chunk the document based on the estimated number of tokens available for the code
        docs = self.file_handler.chunk_document(file_full_path, code, code_token_size)

        # define the response
        response = None

        # if the document is too small, just run it
        if len(docs) == 1:
            template = self.prompt_handler.get_raw_template(template=0)
            self.load_chain(template=template)
            response = self.llm_chain.run(docs[0].page_content)

        # if the document is too big, chunk it and run it
        elif len(docs) > 1:
            template = self.prompt_handler.get_raw_template(template=1)
            self.load_chain(template=template, requires_memory=True)

            responses = []

            for doc in docs:
                response = self.llm_chain.run(doc.page_content)
                responses.append(response)

            # combine the responses and save them
            template = self.prompt_handler.get_raw_template(template=2)
            self.load_chain(template=template, requires_memory=True)

            response = self.llm_chain.run(responses)

        # print(response)
        
        return self._check_response(response)
    
    def _check_response(self, response: str) -> str:

        """
            Checks the response of the langchain model.

            Args:
            ----------
            response: str
                The response to check.
            
            Returns:
            ----------
            str
                The checked response.
        """

        template = self.prompt_handler.get_raw_template(template=4)
        self.load_chain(template=template)
        response = self.llm_chain.run(response)

        response_json = json.loads(response)
        return response_json


    def generate_explaination_for_directory(self, directory_contents: str) -> str:
        """
            Generates an explaination for a directory.
            It is used to generate explainations for the folders in the project.

            Args:
            ----------
            directory_contents: str
                The contents of the directory.
            
            Returns:
            ----------
            str
                The explaination for the directory.
        """
        template = self.prompt_handler.get_raw_template(template=5)
        self.load_chain(template=template)
        response = self.llm_chain.run(directory_contents)

        # print(response)
        return response

    
    def generate_cohesion_coupling_analysis(self, json_report: str) -> str:

        """
            Generates a cohesion and coupling analysis for a project.
            It is used to generate cohesion and coupling analysis for a project.

            Args:
            ----------
            json_report: str
                The json report of the project.
            
            Returns:
            ----------
            str
                The cohesion and coupling analysis for the project.
        """

        template = self.prompt_handler.get_raw_template(template=3)
        prompt = self.prompt_handler.get_prompt(template=3, json_reports=json_report)
        prompt_len = self.prompt_handler.get_prompt_token_lenght(prompt)

        # if the prompt is really big (bigger than the context window size) then load 
        # the gpt which has a bigger context window size

        if prompt_len > self.context_window_size:
            # print("prompt is too big, loading gpt-3.5-turbo-16k")
            self.options["model_name"] = "gpt-3.5-turbo-16k"
            self.load_model()
            self.set_context_window_size(16e3)

            self.load_chain(template=template)
            response = self.llm_chain.run(prompt)
        
        else:
            # print("prompt is not too big, loading gpt-3.5-turbo-16k")
            self.options["model_name"] = "gpt-3.5-turbo-16k"
            self.load_model()
            self.set_context_window_size(16e3)

            self.load_chain(template=template)
            response = self.llm_chain.run(prompt)
        
        #before returning the response, go back to the original cheaper model
        self.options["model_name"] = "gpt-3.5-turbo-16k"
        self.set_context_window_size(16e3)

        self.load_model()

        response_json = json.loads(response)
        return response_json
        


def default_llm(projects_path : str):
    """
        Returns the default langchain model.
        It is used to generate explainations for the files in the project.

        Args:
        ----------
        projects_path: str
            The path to the projects.
        
        Returns:
        ----------
        LLM
            The default langchain model.
    """    

    average_number_of_tokens_per_sentence = 27
    desired_number_of_sentences_per_file = 30
    max_tokens = desired_number_of_sentences_per_file * average_number_of_tokens_per_sentence
    context_window_size = 16e3

    model_name = DEFAULT_LLM

    llm = LLM(
        projects_path=projects_path,
        options={
            "openai_api_key": OPEN_AI_API_KEY,
            "model_name": model_name,
            "temperature": 0,
            "max_tokens": max_tokens,
            "presence_penalty": 2,
            "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
            "verbose": False
        },

    )

    llm.set_context_window_size(context_window_size)
    return llm