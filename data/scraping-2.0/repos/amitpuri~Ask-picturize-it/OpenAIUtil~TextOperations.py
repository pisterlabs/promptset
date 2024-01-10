from OpenAIUtil.Operations import *
import openai

class TextOperations(Operations):
    def set_openai_api_key(self, api_key: str):
        self.deployment_name = None
        self.api_key = api_key
        
    def set_model_name(self, model_name: str):
        self.model_name = model_name

    def set_org_id(self, org_id: str):
        self.org_id = org_id
        
    def set_azure_openai_api_key(self, azure_openai_key: str, azure_openai_endpoint: str, azure_openai_deployment_name: str):
        self.deployment_name = azure_openai_deployment_name        
        self.api_base = azure_openai_endpoint
        self.api_key = azure_openai_key

    def __init__(self):
        self.models = ["gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]
        self.deployment_name = None
        self.org_id = None
        #default to gpt-4
        self.model_name="gpt-4"
        
    def summarize(self, prompt: str):                    
        return self.chat_completion(f"Summarize the following text :{prompt}")

    def chat_completion(self, prompt: str, 
                        system_prompt: str = None, 
                        assistant_prompt: str = None, 
                        temperature=0.7, 
                        max_tokens=800,
                        top_p=0.95,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None):
        try:
            if prompt is not None:
                if system_prompt is None:
                    messages = [{"role": "user", "content": f"{prompt}"}]
                else:
                    messages = [
                        {"role": "user", "content": f"{prompt}"},
                        {"role": "system", "content": f"{system_prompt}"},
                        {"role": "assistant", "content": f"{assistant_prompt}"}
                    ]            
                if self.deployment_name:
                    openai.api_type = "azure"
                    openai.api_version = "2023-05-15" 
                    openai.api_base = self.api_base
                    openai.api_key = self.api_key
                    completion = openai.ChatCompletion.create(
                        engine=f"{self.deployment_name}",
                        messages = messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=stop)                   
                    response = completion["choices"][0]["message"].content                      
                    return "Response from Azure OpenAI", response
                else:
                    openai.api_type = "openai"
                    openai.api_key = self.api_key
                    openai.api_version = '2020-11-07'                    
                    openai.api_base = "https://api.openai.com/v1"
                    if self.org_id:
                        openai.organization = self.org_id
                    if self.model_name in self.models:                                                           
                        completion = openai.ChatCompletion.create(                                  
                                          model=f"{self.model_name}",
                                          messages = messages,
                                          temperature=temperature,
                                          max_tokens=max_tokens,
                                          top_p=top_p,
                                          frequency_penalty=frequency_penalty,
                                          presence_penalty=presence_penalty,
                                          stop=stop)
                        response = completion["choices"][0]["message"].content     
                    else:                                                        

                        completion = openai.Completion.create(
                                      model=f"{self.model_name}",
                                      prompt=f"{prompt}",
                                      temperature=temperature,
                                      max_tokens=max_tokens,
                                      top_p=top_p,
                                      frequency_penalty=frequency_penalty,
                                      presence_penalty=presence_penalty,
                                      stop=stop)
                        response = completion["choices"][0]["text"]
                    return f"Response from OpenAI model {self.model_name}", response                   
                

        except openai.error.OpenAIError as error_except:            
            print(f"TextOperations TextCompletion exception openai.error.OpenAIError, Error {error_except} {openai.api_base} ")
            print(error_except.http_status)
            print(error_except.error)
            return error_except.error["message"], ""                
        except openai.error.APIError as error_except:
            print(f"TextOperations TextCompletion exception openai.error.APIError, Error {error_except}")
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {error_except} {openai.api_base}")
            return error_except.error["message"], ""
        except openai.error.AuthenticationError as error_except:
            print(f"TextOperations TextCompletion exception openai.error.AuthenticationError, Error {error_except}")
            # Handle Authentication error here, e.g. invalid API key
            print(f"OpenAI API returned an Authentication Error: {error_except} {openai.api_base}")        
            return error_except.error["message"], ""
        except openai.error.APIConnectionError as error_except:
            print(f"TextOperations TextCompletion exception openai.error.APIConnectionError, Error {error_except}")
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {error_except} {openai.api_base}")       
            return error_except.error["message"], ""
        except openai.error.InvalidRequestError as error_except:
            print(f"TextOperations TextCompletion exception openai.error.InvalidRequestError, Error {error_except}")
            # Handle connection error here
            print(f"Invalid Request Error: {error_except} {openai.api_base}")    
            return error_except.error["message"], ""
        except openai.error.RateLimitError as error_except:
            print(f"TextOperations TextCompletion exception openai.error.RateLimitError, Error {error_except}")
            # Handle rate limit error
            print(f"OpenAI API request exceeded rate limit: {error_except} {openai.api_base}")
            return error_except.error["message"], ""
        except openai.error.ServiceUnavailableError as error_except:
            print(f"TextOperations TextCompletion exception openai.error.ServiceUnavailableError, Error {error_except}")
            # Handle Service Unavailable error
            print(f"Service Unavailable: {error_except} {openai.api_base}")      
            return error_except.error["message"], ""
        except openai.error.Timeout as error_except:
            print(f"TextOperations TextCompletion exception openai.error.Timeout, Error {error_except} ")
            # Handle request timeout
            print(f"Request timed out: {error_except} {openai.api_base}")
            return error_except.error["message"], ""

    def model_list(self):
        openai.api_key = self.api_key
        openai.api_version = '2020-11-07'
        response = openai.Model.list()
        return [model["id"] for model in response["data"]]
