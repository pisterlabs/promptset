import json
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain

class FlashCardGenerator:
    def __init__(self, subscription_key, endpoint, deployment_name):
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_version="2023-05-15",
            openai_api_key=subscription_key,
            # openai_api_base=endpoint,
            # azure_deployment=deployment_name,
            # openai_api_type="azure",
        )

    def generate_flashcards(self):
        loader = TextLoader("output.txt", encoding='utf-8').load()
        answer = None

        print(loader)

        try:
            chain = load_qa_chain(llm=self.llm, chain_type="map_reduce")
            query = 'output : short questions and short answers in [{"question" : "question 1", "answer" : "answer to question 1"}, {...}] format'
            response = chain.run(input_documents=loader, question=query)

            print(response)
            answer = json.loads(response)

        except Exception as e:
            print(e)
            answer = []

        return answer
    
    def generate_summary(self):
        loader = TextLoader("output.txt", encoding='utf-8').load()
        
        try:
            chain = load_summarize_chain(llm=self.llm, chain_type="map_reduce")
            response = chain.run(input_documents=loader)

            answer = response
        except Exception as e:
            print(e)
            answer = ""

        return answer