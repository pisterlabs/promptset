import argparse

from llama_index import GPTVectorStoreIndex, LLMPredictor, ServiceContext, SimpleDirectoryReader, QuestionAnswerPrompt
from rich.console import Console
from document_search_backend import DocumentSearchBackend
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI


class DocumentSearch:
    def __init__(self, llm, folder_path):
        self.llm = llm
        self.folder_path = folder_path
        self.backend = DocumentSearchBackend()
        self.console = Console()
        self.QA_PROMPT = QuestionAnswerPrompt(
            "You are an AI assistant here to help. Use the following context snippet to answer the question at the end.\n"
            "If you don't know the answer, simply say that you don't know. DO NOT attempt to make up an answer.\n"
            "If the question is not related to the context, politely state that you are only set to answer context-related questions.\n"
            "If the question is not available, answer by saying that the answer is not available in the document data.\n"
            "\n{context_str}\n"
            "Question: {query_str}\n"
            "Helpful Answer:\n\n"
        )

    def search_documents(self, question):
        keywords = self.backend.get_keywords(question)
        self.console.print("Found keywords:", keywords)

        documents = self.backend.search_documents(question)
        for document in documents:
            self.console.print(f"Found top documents: {document['name']} with score {document['score']}")

        return documents

    def query_engine(self, documents, question):
        llm_predictor = LLMPredictor(llm=self.llm)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=3000)
        load_docs = SimpleDirectoryReader(input_files=[i["location"] for i in documents]).load_data()
        index = GPTVectorStoreIndex.from_documents(documents=load_docs, service_context=service_context)
        query_engine = index.as_query_engine(text_qa_template=self.QA_PROMPT, streaming=False)
        return query_engine.query(question)

    def search(self, question):
        documents = self.search_documents(question)
        result = self.query_engine(documents, question)
        return result


class Models:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def instruct_gpt(self, model: str = "text-davinci-003", temperature: float = 0.0):
        return OpenAI(
            openai_api_key=self.api_key,
            model=model,
            temperature=temperature
        )

    def chat_gpt(self, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
        return ChatOpenAI(
            openai_api_key=self.api_key,
            model=model,
            temperature=temperature
        )

def create_gpt_models(api_key):
    model_choices = {
        "davinci": "text-davinci-003",
        "chat": "gpt-3.5-turbo"
    }

    parser = argparse.ArgumentParser(description="Model selection for GPT functions")
    parser.add_argument("--model", type=str, default="chat", choices=model_choices.keys(), help="The GPT model to use: 'davinci' or 'chat'. Default is 'chat'.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature value for response generation")
    args = parser.parse_args()

    models = Models(api_key=api_key)
    selected_model = model_choices[args.model]
    if args.model == "best":
        return models.instruct_gpt(model=selected_model, temperature=args.temperature)
    elif args.model == "chat":
        return models.chat_gpt(model=selected_model, temperature=args.temperature)
    else:
        print("Error: Invalid model specified. Please choose between 'davinci' and 'chat'.")
