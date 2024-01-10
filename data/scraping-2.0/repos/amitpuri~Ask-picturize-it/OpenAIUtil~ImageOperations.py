from OpenAIUtil.Operations import *
import openai
from Utils.ImageUtils import *

class ImageOperations(Operations):
    def set_openai_api_key(self, api_key: str):        
        self.api_key = api_key 
        self.api_base = "https://api.openai.com/v1"          
        self.deployment_name = None
                
    def set_org_id(self, org_id: str):
        self.org_id = org_id
        
    def set_azure_openai_api_key(self, azure_openai_key: str, azure_openai_endpoint: str, azure_openai_deployment_name: str):
        self.api_key = azure_openai_key
        self.api_base = azure_openai_endpoint
        self.deployment_name = azure_openai_deployment_name            
   
    def __init__(self):        
        self.image_utils = ImageUtils()        
        self.org_id = None
   
    def create_variation_from_image(self, picture_file: str, imagesize: str, num_images: int):
        label_inference_variation = "Switch to Output tab to review it"
        if picture_file is None or self.api_key is None:
            return label_inference_variation, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        if not imagesize:  # defaulting to this image size
            imagesize = "256x256"
        try:
            with open(picture_file, "rb") as file_path:
                image = Image.open(file_path)
                width, height = 256, 256
                image = image.resize((width, height))
                byte_stream = BytesIO()
                image.save(byte_stream, format='PNG')
                image_byte_array = byte_stream.getvalue()
        except Exception as err:
            print(f"ImageOperations create_variation_from_image {err}")
        try:
            if self.deployment_name:            
                openai.api_type = "azure"        
                openai.api_version = "2023-05-15" 
                openai.api_base = self.api_base
                openai.deployment_name = self.deployment_name  
            else:
                openai.api_type = "openai"        
                openai.api_version = '2020-11-07'
                openai.api_key = self.api_key
                if self.org_id: 
                    openai.organization = self.org_id
            response = openai.Image.create_variation(
                image=image_byte_array,
                n=num_images,
                size=imagesize
            )
            image_url = response['data'][0]['url']
            try: 
                image_filename = self.image_utils.url_to_image(image_url)
                images_list = self.image_utils.parse_image_name(response['data'])
                return label_inference_variation, image_filename, images_list
            except Exception as err:
                print(f"ImageOperations create_variation_from_image returning output - {err}")
        except openai.error.OpenAIError as error_except:            
            print(f"ImageOperations create_variation_from_image exception openai.error.OpenAIError, Error {error_except} {openai.api_base} ")
            print(error_except.http_status)
            print(error_except.error)
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()            
        except openai.error.APIError as error_except:
            print(f"ImageOperations create_variation_from_image exception openai.error.APIError, Error {error_except}")
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {error_except} {openai.api_base}")
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.AuthenticationError as error_except:
            print(f"ImageOperations create_variation_from_image exception openai.error.AuthenticationError, Error {error_except}")
            # Handle Authentication error here, e.g. invalid API key
            print(f"OpenAI API returned an Authentication Error: {error_except} {openai.api_base}")        
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.APIConnectionError as error_except:
            print(f"ImageOperations create_variation_from_image exception openai.error.APIConnectionError, Error {error_except}")
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {error_except} {openai.api_base}")       
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.InvalidRequestError as error_except:
            print(f"ImageOperations create_variation_from_image exception openai.error.InvalidRequestError, Error {error_except}")
            # Handle connection error here
            print(f"Invalid Request Error: {error_except} {openai.api_base}")    
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.RateLimitError as error_except:
            print(f"ImageOperations create_variation_from_image exception openai.error.RateLimitError, Error {error_except}")
            # Handle rate limit error
            print(f"OpenAI API request exceeded rate limit: {error_except} {openai.api_base}")
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.ServiceUnavailableError as error_except:
            print(f"ImageOperations create_variation_from_image exception openai.error.ServiceUnavailableError, Error {error_except}")
            # Handle Service Unavailable error
            print(f"Service Unavailable: {error_except} {openai.api_base}")      
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.Timeout as error_except:
            print(f"ImageOperations create_variation_from_image exception openai.error.Timeout, Error {error_except} ")
            # Handle request timeout
            print(f"Request timed out: {error_except} {openai.api_base}")
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
            
        

    def create_image_from_prompt(self, prompt: str, imagesize: str, num_images: int):
        label_picturize_it = "Switch to Output tab to review it"
        if prompt is None or self.api_key is None:
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        if not imagesize:
            imagesize = "256x256"
        try:
            if self.deployment_name:            
                openai.api_type = "azure"        
                openai.api_version = "2023-05-15" 
                openai.api_base = self.api_base
                openai.deployment_name = self.deployment_name  
            else:
                openai.api_type = "openai"        
                openai.api_version = '2020-11-07'
                openai.api_key = self.api_key
                if self.org_id: 
                    openai.organization = self.org_id
            
            response = openai.Image.create(
                    prompt=prompt,
                    n=num_images,
                    size=imagesize)
            image_url = response['data'][0]['url']
            return label_picturize_it, self.image_utils.url_to_image(image_url), self.image_utils.parse_image_name(response['data'])
        except openai.error.OpenAIError as error_except:            
            print(f"ImageOperations create_image_from_prompt exception openai.error.OpenAIError, Error {error_except} {openai.api_base} ")
            print(error_except.http_status)
            print(error_except.error)
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()            
        except openai.error.APIError as error_except:
            print(f"ImageOperations create_image_from_prompt exception openai.error.APIError, Error {error_except}")
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {error_except} {openai.api_base}")
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.AuthenticationError as error_except:
            print(f"ImageOperations create_image_from_prompt exception openai.error.AuthenticationError, Error {error_except}")
            # Handle Authentication error here, e.g. invalid API key
            print(f"OpenAI API returned an Authentication Error: {error_except} {openai.api_base}")        
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.APIConnectionError as error_except:
            print(f"ImageOperations create_image_from_prompt exception openai.error.APIConnectionError, Error {error_except}")
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {error_except} {openai.api_base}")       
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.InvalidRequestError as error_except:
            print(f"ImageOperations create_image_from_prompt exception openai.error.InvalidRequestError, Error {error_except}")
            # Handle connection error here
            print(f"Invalid Request Error: {error_except} {openai.api_base}")    
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.RateLimitError as error_except:
            print(f"ImageOperations create_image_from_prompt exception openai.error.RateLimitError, Error {error_except}")
            # Handle rate limit error
            print(f"OpenAI API request exceeded rate limit: {error_except} {openai.api_base}")
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.ServiceUnavailableError as error_except:
            print(f"ImageOperations create_image_from_prompt exception openai.error.ServiceUnavailableError, Error {error_except}")
            # Handle Service Unavailable error
            print(f"Service Unavailable: {error_except} {openai.api_base}")      
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        except openai.error.Timeout as error_except:
            print(f"ImageOperations create_image_from_prompt exception openai.error.Timeout, Error {error_except} ")
            # Handle request timeout
            print(f"Request timed out: {error_except} {openai.api_base}")
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()

        