import os
import requests
from typing import Any, Dict, List, Tuple, Union
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from agent.utils.configuration import load_config
from agent.utils.utility import generate_prompt
from langchain.docstore.document import Document as LangchainDocument
from agent.utils.utility import replace_multiple_whitespaces
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import ChatOllama
from agent.backend.llm_services.LLM import BaseLLM  


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


class OllamaLLM(BaseLLM):
    def __init__(self):
        self.channeling_system_message = os.getenv("CHANNELING_SYSTEM_MESSAGE",default="Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworte diese aus den Quellen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist! USER: BEGININPUT")
        self.q_and_a_system_message = os.getenv("Q_A_SYSTEM_MESSAGE",default="Du bist ein ehrlicher, respektvoller und ehrlicher Assistent. Zur Beantwortung der Frage nutzt du nur den Text, welcher zwischen <INPUT> und </INPUT> steht! Findest du keine Informationen im bereitgestellten Text, so antwortest du mit 'Ich habe dazu keine Informationen'")
            
    def send_chat_completion(self, text: str, query: str, conversation_type: str, messages: any) -> str:
        """Sent completion request to ollama API.

        Args:
            text (str): The text on which the completion should be based.
            query (str): The query for the completion.


        Returns:
            str: Response from the the model.
        """

        if conversation_type == "CHANNELING":
            prompt = generate_prompt(prompt_name=f"{OLLAMA_MODEL}-channeling.j2", text=text, query=query, system=self.channeling_system_message, language="de")
        else:
            prompt = generate_prompt(prompt_name=f"{OLLAMA_MODEL}-qa.j2", text=text, query=query, system=self.q_and_a_system_message, language="de")

        messages.append({"role": "user", "content": prompt})
        logger.debug(f"Filled prompt before request: {prompt}")

        
        ollama_model = f"{os.environ.get('OLLAMA_MODEL')}:{os.environ.get('OLLAMA_MODEL_VERSION')}" if os.environ.get('OLLAMA_MODEL_VERSION') else os.environ.get('OLLAMA_MODEL')

        #LangChain message list for a future feature (message history)
        messagesBaseFormat: List[BaseMessage] = [HumanMessage(content=m["content"], additional_kwargs={}) if m["role"] == "user"
                        else AIMessage(content=m["content"], additional_kwargs={}) if m["role"] == "assistant"
                        else SystemMessage(content=m["content"], additional_kwargs={})
                        for m in messages]

        response = self.generate_request(url_ollama_generateEndpoint=f"http://{OLLAMA_URL}:{OLLAMA_PORT}/api/generate",
                                        model=ollama_model,
                                        full_prompt=prompt)
        
        return response

    def chat(self, documents: list[tuple[LangchainDocument, float]], messages: any, query: str, conversation_type: str, summarization: bool = False) -> Tuple[str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]:
        """Takes a list of documents and returns a list of answers.

        Args:
            documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
            query (str): The query to ask.
            summarization (bool, optional): Whether to use summarization. Defaults to False.

        Returns:
            Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
        """
        text = ""
        if conversation_type == "Q_AND_A":
            # extract the texts and meta data from the documents
            texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
            text = " ".join(texts)
            meta_data = [doc.metadata for doc in documents]
        else:
            # different logic in case of channeling (if needed)
            texts = [replace_multiple_whitespaces(doc.page_content) for doc in documents]
            text = " ".join(texts)
            meta_data = [doc.metadata for doc in documents]
        

        answer=""
        try:
            # fills the prompt, sends a request and returns the response
            answer = self.send_chat_completion(text=text, query=query, conversation_type=conversation_type, messages=messages)

        except ValueError as e:
            logger.error("Error found:")
            logger.error(e)
            answer = "Error while processing the completion"
        logger.debug(f"LLM response: {answer}")
        
        return answer, meta_data


    def generate_request(self, url_ollama_generateEndpoint: str, model: str, full_prompt: str):
        """Generates a request to ollama.

        Args:
            url_ollama_generateEndpoint (str): A string url to the endpoint of ollama generate api.
            model (str): The model name.
            full_prompt (string): The whole prompt with everything included.
            raw_mode (str): A string which could either be True or False to enable raw mode for the ollama request.

        Returns:
            str: LLM Response
        """
        url = url_ollama_generateEndpoint

        logger.info(f"URL: {url}, Model: {model}")

        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "prompt": full_prompt,
            "raw": True,
            "stream": False,
            "options": {"stop": ["<|im_start|>", "<|im_end|>"]}
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            logger.debug("HTTP status code 200: Request was successful!")
        else:
            logger.debug(f"Error {response.status_code}: {response.text}")

        return response.json()["response"]

