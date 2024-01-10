from langchain.chat_models import ChatOpenAI
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import boto3
if True:
    import sys
    sys.path.append("../")
# from src.utils.config import load_config
from src.utils.config import load_config
from src.utils.logger import Logger

# set logger
logger = Logger(__name__).get_logger()

class BaseLLMException(Exception):
    """Custom class for handling BaseLLM Exceptions"""

class BaseLLM:
    """Base LLM"""

    def __init__(self, debug=False, streaming=False, type_model=None) -> None:
        try:
            self.debug = debug
            self.config = load_config(debug=self.debug)
            self.streaming = streaming

            # get type llm
            self.type_llm = self.get_type_llm(type_model=type_model)

            # Instance LLM
            self.llm = self.instance_model()
        except Exception as error:
            logger.error(f"Error in BaseLLM: {error}")
            raise BaseException(
                f"Exception caught in BaseLLM Module - init: {error}"
            )

    def __call__(self):
        """Return llm"""
        try:
            return self.llm
        except Exception as error:
            raise BaseLLMException(
                f"Exception caught in BaseLLM Module - __call__: {error}"
            )

    def get_type_llm(self, type_model):
        """Return type llm"""
        if type_model == None:
            type_llm = self.config['base_llm']['type_llm']
        else:
            type_llm = type_model
        return type_llm

    def instance_model(self):
        """Return instantiated llm"""
        try:
            if self.streaming:
                callbacks = [StreamingStdOutCallbackHandler()]
            else:
                callbacks = None

            if self.type_llm == "openai":
                # Instance LLM
                llm = ChatOpenAI(
                    temperature=self.config['base_llm']['temperature'],
                    model_name=self.config['openai_llm']['model_name'],
                    streaming=self.streaming,
                    model_kwargs={'top_p': 0.09},
                    callbacks=callbacks
                )

            if self.type_llm == "bedrock":
                # creat conection python to AWS
                bedrock_client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.config['bedrock_llm']['region_name'],
                )
                # Instance LLM
                llm = Bedrock(
                    model_id=self.config['bedrock_llm']['model_name'],
                    client=bedrock_client,
                    model_kwargs={
                        "temperature": self.config['base_llm']['temperature'],
                        "topP": self.config['bedrock_llm']['topP'],
                        "maxTokens": self.config['bedrock_llm']['maxTokens']
                    },
                    callbacks=callbacks
                )
        except Exception as e:
            logger.error(f"Error in instance_model: {e}")
            raise e
        return llm


if __name__ == "__main__":
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from promptwatch import PromptWatch

    base_llm = BaseLLM(debug=True, streaming=True, type_model="openai")
    query = "create history about a wizard"

    prompt = PromptTemplate(
        template="""You are an assiant,
        {query}""",
        input_variables=["query"],
    )

    llm_chain = LLMChain(
        llm=base_llm(), prompt=prompt)

    # with PromptWatch(api_key="") as pw:
    # res = base_llm.llm.predict(query)
    # print(res)

    response = llm_chain.run(query=query)
    print(f"response: {response}")

    with PromptWatch(api_key="") as pw:
        res = base_llm.llm.predict(query)
        print(res)
