from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain import PromptTemplate
from langchain.chains import LLMChain
import os
from os.path import dirname, join 
from dotenv import load_dotenv
from .vecdb import Embedder 

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
API_TOKEN = os.environ.get("APIKEY")
PROJECT_ID = os.environ.get("PROJECT_ID")

class NotesAnalyst():
    def __init__(self):
        self.template = 'ACT LIKE A SURVEY ANALYST MODEL. YOU RECEIVE SURVEY QUESTIONS AND THE ANSWERS THE USER PROVIDED. SUMMARIZE THE ANSWERS AND CHARACTERIZE THE USERS. SURVEY AND ANSWERS: {note} SUMMARIZATION AND CHARACTERIZATION:'
        self.prompt = PromptTemplate(
        input_variables=["note"],
        template=self.template)

        GenParams().get_example_values()

        #model hyperparameters 
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 10,
            GenParams.MAX_NEW_TOKENS: 50,
            GenParams.TEMPERATURE: 0.0,
            GenParams.REPETITION_PENALTY: 1,
            GenParams.LENGTH_PENALTY: {'decay_factor': 2, 'start_index': 20}
        }

        #initializing the model 
        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": API_TOKEN,  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id=PROJECT_ID
        )

        self.chain = LLMChain(llm=self.model.to_langchain(), prompt=self.prompt, verbose=False)
        self.embedder = Embedder()

    def analyze(self, note:str):
        analysis = self.chain(note)

        return analysis['text']

# if __name__ == '__main__':
#     analyst  = NotesAnalyst()
#     note = open('test.txt', 'r').read()
#     note=note.replace('\n', '')
#     analysis = analyst.analyze(note)['text']
#     print(analysis)