import os 
import pickle 
import textwrap 
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

from PyPDF2 import PdfReader
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

# import from files 
from src.prompt_storage import PromptStorageForChains

class BaseChain:
    def __init__(self, project_name : str, artifact_folder : str, source_path : str, openai_key : Optional[str] = None, openai_temp : int = 0) -> None:
        """
        Args:
            :param project_name : (str) The name of the project, with this name all the artifacts will be generated 
            :param artifact_folder : (str) The root locations where all the artifacts will be stored
            :param openai_key : (str) The open ai key, 
            :param openai_temp : (int) The temperature of the open ai model, recommeded value: 0
        """
        self.project_name = project_name
        self.artifact_folder = artifact_folder
        self.source_path = source_path
        self.openai_key = openai_key
        self.openai_temp = openai_temp
        self.prompt = PromptStorageForChains()

        load_dotenv(dotenv_path='.env/openai.env')
        # FIXME: Should also hold for AWS  S3 bucket
        self.artifact_path = os.path.join(self.artifact_folder, self.project_name)

        # Loading the LLM 
        try:
            self.openai_key = os.getenv("OPENAI_API_KEY") if self.openai_key is None else self.openai_key
            self.llm = OpenAI(
                openai_api_key=self.openai_key, temperature=self.openai_temp
            )
        except Exception as e:
            print(f"Open AI Exception occured at: {e}")
            return None
        
        # Creating the artifact folder inside the provided folder location 
        try:
            if os.path.exists(self.artifact_path):
                print(f"Artifact folder already exists at {self.artifact_path}")
            else:
                if not os.path.exists(self.artifact_folder):
                    os.mkdir(self.artifact_folder)
                
                if not os.path.exists(self.artifact_path):
                    os.mkdir(self.artifact_path)
                print("=> Artifact folder created successfully")
        except Exception as e:
            print(f"Exception occured at: {e}")
            return None
        
        # Loading the document
        try:
            self._document_loader = PdfReader(self.source_path)
            self.pages = self._document_loader.pages

            print(self.pages)
            self.doc_texts = []
            
            for text in self.pages: self.doc_texts.append(text.extract_text(0))
            self.docs = [Document(page_content=t) for t in self.doc_texts]

        except Exception as e:
            print(f"Document Exception occured at: {e}")
            return None
        print("=> LLM and target document loaded successfully")


class SummarizeChain(BaseChain):
    def __init__(self, project_name : str, artifact_folder : str, source_path : str, openai_key : Optional[str] = None, openai_temp : int = 0) -> None:
        super().__init__(project_name, artifact_folder, source_path, openai_key, openai_temp)
    
    def run_story_summary_chain(self) -> Dict[str, Any]:
        self.base_prompt_template, self.refined_prompt_template = self.prompt.fetch_summarize_prompt_template()
        
        # FIXME: Right now the prompt templates has hardcoded inputs we need to provide it in args 
        # for newer versions 

        self.BASE_PROMPT = PromptTemplate(
            template = self.base_prompt_template, 
            input_variables=["text"]
        )

        self.REFINED_PROMPT = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=self.refined_prompt_template,
        )

        self.summarizer_chain = load_summarize_chain(
            self.llm, 
            chain_type="refine", # this is by defaiult in this version 
            return_intermediate_steps=True,
            question_prompt=self.BASE_PROMPT,
            refine_prompt=self.REFINED_PROMPT, verbose=True)
        
        # Doing a sanity check if the results are aleady present or not 
        if os.path.exists(os.path.join(self.artifact_path, "story_summary.pkl")):
            print("=> Story summary already exists, loading the summary")
            with open(os.path.join(self.artifact_path, "story_summary.pkl"), "rb") as f:
                return pickle.load(f)
        else:
            self.output_summary = self.summarizer_chain({
                "input_documents": self.docs
            }, return_only_outputs=True)

            wrapped_text = textwrap.fill(self.output_summary['output_text'], 
                                         width=100,break_long_words=False,
                                         replace_whitespace=False)
            intermediate_steps = []
            for step in self.output_summary['intermediate_steps']:
                intermediate_steps.append(step)
            
            response_dict = {
                'full_summary' : wrapped_text,
                "summary_chunks" : intermediate_steps 
            }

            with open(os.path.join(self.artifact_path, "story_summary.pkl"), "wb") as f:
                pickle.dump(response_dict, f)
            
            return response_dict


class QuestionAnswerChain(BaseChain):
    def __init__(self, project_name : str, artifact_folder : str, openai_key : Optional[str] = None, openai_temp : int = 0) -> None:
        super().__init__(project_name, artifact_folder, openai_key, openai_temp)
    
    def run_question_answer_chain(self) -> Dict[str, Any]:
        raise NotImplementedError("Question Answer Chain is not implemented yet")


 # GraphViz Chain is not implemented yet
 #    
class GraphVizChain(BaseChain):
    def __init__(self, project_name : str, artifact_folder : str, openai_key : Optional[str] = None, openai_temp : int = 0) -> None:
        super().__init__(project_name, artifact_folder, openai_key, openai_temp)
    
    def run_graphviz_chain(self) -> Dict[str, Any]:
        raise NotImplementedError("GraphViz Chain is not implemented yet")