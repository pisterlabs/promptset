# state.py
import reflex as rx
import os

from dotenv import load_dotenv

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.memory import ConversationBufferMemory
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain.chains import LLMChain



class State(rx.State):

    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: str

    async def handle_upload(
        self, files: list[rx.UploadFile]
    ):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_asset_path(file.filename)

            # Save the file.
            with open(outfile, "wb") as file_object:
                file_object.write(upload_data)

            # Update the img var.
            self.img.append(file.filename)
            #print(self.data)
            
    #config Watsonx.ai environment
    load_dotenv()
    api_key = os.getenv("API_KEY", None)
    ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
    project_id = os.getenv("PROJECT_ID", None)
    if api_key is None or ibm_cloud_url is None or project_id is None:
        raise Exception("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
    else:
        creds = {
            "url": ibm_cloud_url,
            "apikey": api_key 
        }

    def send_to_watsonxai(self):

        #assert not any(map(lambda prompt: len(prompt) < 1, prompts)), "make sure none of the prompts in the inputs prompts are empty"

        # Instantiate parameters for text generation
        model_params = {
            GenParams.DECODING_METHOD: 'sample',
            GenParams.MIN_NEW_TOKENS: 3,
            GenParams.MAX_NEW_TOKENS: 10,
            GenParams.RANDOM_SEED: 42,
            GenParams.TEMPERATURE: .1,
            GenParams.REPETITION_PENALTY: 2.0,
        }


        # Instantiate a model proxy object to send your requests
        model = Model(
            model_id="meta-llama/llama-2-70b-chat",
            params=model_params,
            credentials=self.creds,
            project_id=self.project_id)

        llm = WatsonxLLM(model)
        
        template = """extract the emotions the reviewer expressed return answer as a comma separated list
            Example: Review: I had relatives in the nursing home, which was understaffed, they have multiple needs, and they need a lot of care. They are understaffed and they need more than one staff to assist with caring for the residents. My client has sorosis of the liver, her mind is gone, patience is required to take care of your patient. Output: Sadness, Worry
            Review text: '''{review}'''
            Output:
            """

        prompt = PromptTemplate(
            input_variables=["review"], template=template
        )
        #memory = ConversationBufferMemory(memory_key="chat_history")
        
        llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        #memory=memory,
        )
        #print(memory.history)
        return(llm_chain.predict(review=self.question))

    def answer(self):
        # Our chatbot is not very smart right now...
        self.chat_history = ""
        self.chat_history  = f"Review: {self.question} \n\n\n --------------------Sentiment: {self.send_to_watsonxai()}"
        
        # Clear the question input.
        self.question = ""
        