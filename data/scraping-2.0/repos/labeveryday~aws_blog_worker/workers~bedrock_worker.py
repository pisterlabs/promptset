import boto3
import os
import json
from pprint import pprint


SERVICE_NAME = 'bedrock'
REGION_NAME = 'us-west-2'
ENDPOINT_URL = 'https://bedrock.us-west-2.amazonaws.com'


class BedrockWorker:
    def __init__(self, service_name=SERVICE_NAME, region_name=REGION_NAME, endpoint_url=ENDPOINT_URL):
        """
        Initializes Bedrock API client

        Parameters:
            service_name (str):     AWS service name for Bedrock client
            region_name (str):      AWS region for Bedrock
            endpoint_url (str):     Bedrock endpoint URL
        
        Attributes:
            bedrock (client):       Boto3 Bedrock client
            models (list):          List of available model IDs
            prompt_data (str):      Prompt text for model
            prompt_response (dict): Generated response for prompt
        """
        self.service_name = SERVICE_NAME
        self.region_name = REGION_NAME
        self.endpoint_url = ENDPOINT_URL
        self.bedrock = self.connect()
        self.models = self.__get_models()
        self.prompt_data = None
        self.prompt_response = None
        self.model = None
    
    def connect(self):
        connect = boto3.client(
            service_name=self.service_name, 
            region_name=self.region_name,
            endpoint_url=self.endpoint_url
        )
        return connect
    
    def __get_models(self) -> list:
        """
        Gets list of available model IDs

        Returns:
            list: List containing IDs of available models
        """
        models_list = self.bedrock.list_foundation_models()['modelSummaries']
        models = []
        for model in models_list:
            models.append(model['modelId'])
        return models

    def prompt(self, prompt_data, model_id='anthropic.claude-v2', temperature=1,
               topP=1.0, topk=250, maxTokenCount=4096, stop_sequences=[]):
        """
        Makes API request to generate response for prompt
        Parameters:
        
            prompt_data (str):      The prompt text to generate response for
            model_id (str):         The ID of the AI model to use
            temperature (float):    Sampling temperature for model
            topP (float):           Top p sampling parameter for model
            topk (int):             Top k sampling parameter for model
            maxTokenCount (int):    Maximum number of tokens to generate
            stop_sequences (list):  Stop sequences to prevent model generating
        
        Returns:
            dict: Dictionary containing model_id and generated response body
        """
        if model_id not in self.models:
            raise Exception(f"Model {model_id} does not exist")
        
        self.prompt_data = prompt_data
        self.model = model_id
        payload = {
            "prompt_data": prompt_data,
            "model_id": model_id,
            "temperature": temperature,
            "topP": topP,
            "topk": topk,
            "maxTokenCount": maxTokenCount,
            "stop_sequences": stop_sequences
        }
        if model_id == 'anthropic.claude-v1' or model_id == 'anthropic.claude-v2' or model_id == 'anthropic.claude-instant-v1':
            response_body = self.get_prompt_anthropic(**payload)
            response = {
                "model_id": model_id,
                "response": response_body['completion']
            }
            self.prompt_response = response
            return response
        
        elif model_id == 'amazon.titan-tg1-large' or model_id == 'amazon.titan-e1t-medium':
            response_body = self.get_prompt_amazon(**payload)
            response = {
                "model_id": model_id,
                "response": response_body['results'][0]['outputText']
            }
            return response

    def get_prompt_anthropic(self, prompt_data, model_id='anthropic.claude-v2', temperature = 1,
               topP = 1.0, topk = 250, maxTokenCount = 8191, stop_sequences = []):
        """
        Generates response from Anthropic model for given prompt

        Parameters:
            prompt_data (str):      Prompt text to generate response for
            model_id (str):         Anthropic model ID to use
                                    - anthropic.claude-v1 (12k tokens) - Anthropic's most powerful model
                                    - anthropic.claude-v2 (12k tokens)
                                    - anthropic.claude-instant-v1 (9k tokens)
            temperature (float):    Sampling temperature for model
            topP (float):           Top p sampling parameter
            topk (int):             Top k sampling parameter
            maxTokenCount (int):    Maximum number of tokens to generate
            stop_sequences (list):  Stop sequences to prevent model generating
        
        Returns:
            dict: API response JSON containing generated text
        """
        models = ['anthropic.claude-v1', 'anthropic.claude-v2', 'anthropic.claude-instant-v1']
        try:
            if model_id not in models:
                raise Exception(f"Model {model_id} is not an anthropic model.")
            payload = {
                "body": json.dumps(
                    {
                        "prompt": f"\n\nHuman: {prompt_data}\nAssistant:",
                        "max_tokens_to_sample": maxTokenCount,
                        "temperature": temperature,
                        "top_k": topk,
                        "top_p": topP,
                        "stop_sequences":[]
                    }
                    ),
                "modelId":  model_id,
                "accept": '*/*',
                "contentType": 'application/json'
                }
                
            response = self.bedrock.invoke_model(**payload)
            response_body = json.loads(response.get('body').read())
            return response_body
        except Exception as e:
            print(e)
            exit(1)
    
    def get_prompt_amazon(self, prompt_data, model_id='anthropic.claude-v2', temperature = 1,
               topP = 1.0, topk = 250, maxTokenCount = 4096, stop_sequences = []):
        """
        Generates response from Amazon Titan model for given prompt

        Parameters:
            prompt_data (str):      Prompt text to generate response for
            model_id (str):         Anthropic model ID to use
                                    - amazon.titan-tg1-large (8k tokens) -  For tasks such as summarization, text generation (for example, creating a blog post), classification, open-ended Q&A, and information extraction.
                                    - amazon.titan-e1t-medium (512 tokens) - Fast and cost-effective.
            temperature (float):    Sampling temperature for model
            topP (float):           Top p sampling parameter
            topk (int):             Top k sampling parameter
            maxTokenCount (int):    Maximum number of tokens to generate
            stop_sequences (list):  Stop sequences to prevent model generating
        
        Returns:
            dict: API response JSON containing generated text
        """
        models = ['amazon.titan-tg1-large', 'amazon.titan-e1t-medium']
        try:
            if model_id not in models:
                raise Exception(f"Model {model_id} is not an amazon titan model.")
            payload = {
                "modelId":  model_id,
                "accept": '*/*',
                "contentType": 'application/json',
                "body": json.dumps(
                    {
                       "inputText": prompt_data,
                       "textGenerationConfig": {
                          "maxTokenCount": maxTokenCount,
                          "stopSequences": stop_sequences,
                          "temperature":temperature,
                          "topP": topP
                         }
                    }
                )
                
            }
            response = self.bedrock.invoke_model(**payload)
            response_body = json.loads(response.get('body').read())
            return response_body
        except Exception as e:
            print(e)
            exit(1)

if __name__ == "__main__":
    from langchain.prompts import PromptTemplate
    
    demo = BedrockWorker()
    
    prompt_data = """Titles for a viral video discussion on AWS Infrastructure security."""
    # test = "Give me a viral tweet that promotes a Youtube video interview on: Will network engineers be replaced by generative AI?"
    print(demo.prompt(prompt_data))