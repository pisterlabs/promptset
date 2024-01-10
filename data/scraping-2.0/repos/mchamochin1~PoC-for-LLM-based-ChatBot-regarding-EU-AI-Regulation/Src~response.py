# A response service
#
# This service will take the output of the information retrieval service and generate
# from it an answer to the user’s question. We again use an OpenAI model to
# generate the answer.


from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from data import DataService
from intent import IntentService

RESPONSE_MODEL_NAME="text-davinci-003"

class ResponseService():
    def __init__(self):
        # Conrains a DataService and IntendtService
        self.data_service = DataService()
        self.intent_service = IntentService()

        self.response_model = RESPONSE_MODEL_NAME
        self.db = self.data_service.vectorialdb()

     
    def generate_response(self, query, streaming_callback):

        # model_kwargs = {
        #     "p": 0.01,
        #     "k": 0,
        #     "stop_sequences": [],
        #     "return_likelihoods": "NONE",
        #     "stream": True
        # }

        # Create the llm model for the response
        self.llm = OpenAI(
            model = self.response_model,
            # model_kwargs=model_kwargs,
            # max_tokens = 4000,
            temperature = 0,
            streaming=True,
            callbacks=[streaming_callback],
            )
        #self.llm = OpenAI(model="gpt-3.5-turbo-instruct")

        # I select the vectorial db for the retrieval
        chain = RetrievalQA.from_llm(llm=self.llm, retriever=self.db.as_retriever())
        # print("Chain = ", chain, "\n")

        # Get the intent for the provided query
        intents = self.intent_service.get_intent(query)
        # print("Intents = ", intents, "\n")

        # return the answer
        result=chain.run(intents)
        # print("Result = ", result, "\n")
        return result
    
