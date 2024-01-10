import os
from CloudinaryUtil.CloudinaryClient import *  #set_folder_name,search_images, upload_image
from MongoUtil.CelebDataClient import * #get_describe, get_celebs_response
from MongoUtil.StateDataClient import *
from MongoUtil.KBDataClient import *
from OpenAIUtil.ImageOperations import * #create_image_from_prompt, create_variation_from_image
from OpenAIUtil.TranscribeOperations import *  #transcribe
from OpenAIUtil.TextOperations import *
from Utils.ImageUtils import * #fallback_image_implement
from Utils.DiffusionImageGenerator import * #generate_image


# Ask me UI Component handler
class AskMeUI:
    def __init__(self):
        self.NO_API_KEY_ERROR="Review Configuration tab for keys/settings"
        self.LABEL_GPT_CELEB_SCREEN = "Name, Describe, Preview and Upload"
        self.image_utils = ImageUtils()
        self.api_key = None         
        self.azure_openai_key = None
        self.azure_openai_deployment_name = None
        self.org_id = None
        self.model_name = None

    def get_private_mongo_config(self):
        return os.getenv("P_MONGODB_URI"), os.getenv("P_MONGODB_DATABASE")
        

    def set_cloudinary_config(self, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret):        
        self.cloudinary_cloud_name = cloudinary_cloud_name
        self.cloudinary_api_key = cloudinary_api_key
        self.cloudinary_api_secret = cloudinary_api_secret

    def set_org_id(self, org_id: str):
        self.org_id = org_id

    def set_model_name(self, model_name: str):
        self.model_name = model_name
        
    def set_openai_config(self, api_key: str):
        self.api_key = api_key                        
        self.azure_openai_key = None
        self.azure_openai_deployment_name = None
        self.azure_openai_api_base="https://api.openai.com"

    def set_azure_openai_config(self, azure_openai_key: str, azure_openai_api_base: str, azure_openai_deployment_name: str):
        self.api_key = None                        
        self.azure_openai_key = azure_openai_key
        self.azure_openai_api_base = azure_openai_api_base
        self.azure_openai_deployment_name = azure_openai_deployment_name
    
    def set_mongodb_config(self, mongo_config, connection_string, database):
        if not mongo_config:
            self.connection_string, self.database = self.get_private_mongo_config()
        else:
            self.connection_string = connection_string
            self.database = database
        
    def get_celebs_response_handler(self, keyword):        
        celeb_client = CelebDataClient(self.connection_string, self.database)
        name, prompt, response, image_url, generated_image_url = celeb_client.get_celebs_response(keyword)
        try:
            if response is not None and len(response)==0:
                response = None
            if prompt is not None and len(prompt)==0:
                prompt = None
            if image_url is not None and len(image_url)==0:
                image_url = None
            if generated_image_url is not None and len(generated_image_url)==0:
                generated_image_url = None

            if image_url and generated_image_url and response and prompt:
                return name, prompt, response, self.image_utils.url_to_image(image_url), self.image_utils.url_to_image(generated_image_url)
            elif image_url is None and generated_image_url is None and response and prompt:
                return name, prompt, response, None, None            
            elif response and prompt and (image_url and not generated_image_url):
                return name, prompt, response, self.image_utils.url_to_image(image_url), None
            elif response and prompt and (not image_url and generated_image_url):
                return name, prompt, response, None, self.image_utils.url_to_image(generated_image_url)
            elif not response and not prompt and not image_url and generated_image_url:
                return keyword, "", "", None, None
        except Exception as err:
            return f"Error {err} in AskMeUIHandlers -> Celebs Response Handler", "", "", None, None
    
    
            
    def cloudinary_search(self, folder_name):
        if not self.cloudinary_cloud_name:
            return
        if not self.cloudinary_api_key:
            return
        if not self.cloudinary_api_secret:
            return
        if not folder_name:
            return
    
        cloudinary_client = CloudinaryClient(self.cloudinary_cloud_name, self.cloudinary_api_key, self.cloudinary_api_secret)
        cloudinary_client.set_folder_name(folder_name)
        return cloudinary_client.search_images()
    
            
    def cloudinary_upload(self, folder_name, input_celeb_picture, celebrity_name):
        if not self.cloudinary_cloud_name:
            return "", self.image_utils.fallback_image_implement()
        if not self.cloudinary_api_key:
            return "", self.image_utils.fallback_image_implement()
        if not self.cloudinary_api_secret:
            return "", self.image_utils.fallback_image_implement()
        if not folder_name:
            return "", self.image_utils.fallback_image_implement()
        cloudinary_client = CloudinaryClient(self.cloudinary_cloud_name, self.cloudinary_api_key, self.cloudinary_api_secret)
        cloudinary_client.set_folder_name(folder_name)
        url = cloudinary_client.upload_image(input_celeb_picture, celebrity_name)    
    
        return "Uploaded - Done", self.image_utils.url_to_image(url)
    
    def generate_image_diffusion_handler(self, name, prompt):
        if name:
            try: 
                image_generator = DiffusionImageGenerator()
                output_generated_image = image_generator.generate_image(name, prompt)
                return "Image generated using stabilityai/stable-diffusion-2 model", output_generated_image
            except Exception as err:
                return f"Error : {err}", None
        else:
            return "No Name given", None
            
    def transcribe_handler(self, audio_file, language="en"):
        if not self.api_key:
            return self.NO_API_KEY_ERROR, ""
        if not audio_file:
            return "No audio file", ""
            
        transcribe_operations = TranscribeOperations()
        transcribe_operations.setOpenAIConfig(self.api_key, self.org_id)
        return transcribe_operations.transcribe(audio_file, language)
    
    def create_variation_from_image_handler(self, input_image_variation, input_imagesize, input_num_images):
        if self.api_key is None and self.azure_openai_key is None:
            print("using Diffusion model from api_key and azure_openai_key is set to None")
            diffusion_image_generator = DiffusionImageGenerator()
            label_inference_variation = "Switch to Output tab to review it"
            return label_inference_variation, diffusion_image_generator.image_variation(input_image_variation),""

        
        image_operations = ImageOperations()
        if self.azure_openai_deployment_name:
            image_operations.set_azure_openai_api_key(self.azure_openai_key, self.azure_openai_api_base, self.azure_openai_deployment_name)            
        else:
            image_operations.set_openai_api_key(self.api_key)
            if self.org_id:
                image_operations.set_org_id(self.org_id)
        
        return image_operations.create_variation_from_image(input_image_variation, input_imagesize, input_num_images)
    
    def create_image_from_prompt_handler(self, input_prompt, input_imagesize, input_num_images):
        if self.api_key is None and self.azure_openai_key is None:
            print("exiting from api_key and azure_openai_key is set to None")
            return self.NO_API_KEY_ERROR, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()

        image_operations = ImageOperations()
        if self.azure_openai_deployment_name:
            image_operations.set_azure_openai_api_key(self.azure_openai_key, self.azure_openai_api_base, self.azure_openai_deployment_name)
        else:
            image_operations.set_openai_api_key(self.api_key)
            if self.org_id:
                image_operations.set_org_id(self.org_id)
            
        return image_operations.create_image_from_prompt(input_prompt, input_imagesize, input_num_images)
    
    
    def ask_chatgpt(self, prompt, keyword, prompttype):
        if not prompt or not keyword:
            return "Prompt or keyword is required!",""
            database_prompt, database_response = self.ask_chatgpt_database_response(prompt, keyword, prompttype)
            if database_response:
                return database_prompt, database_response
        try:
            if self.api_key or self.azure_openai_key:
                operations = TextOperations()
                if self.azure_openai_deployment_name:
                    operations.set_azure_openai_api_key(self.azure_openai_key, self.azure_openai_api_base, self.azure_openai_deployment_name)
                else:
                    operations.set_openai_api_key(self.api_key)
                    if self.model_name:
                        operations.set_model_name(self.model_name)
                    if self.org_id:
                        operations.set_org_id(self.org_id)
                        
                response_message, response = operations.chat_completion(prompt)
                state_data_client = StateDataClient(self.connection_string, self.database)
                state_data_client.save_prompt_response(prompt, keyword, response, prompttype)
                return response_message, response
            else:
                return self.NO_API_KEY_ERROR, ""
        except Exception as err:
            return f"Error {err} in AskMeUIHandlers -> ask_chatgpt",""

    def ask_chatgpt_database_response(self, keyword, prompttype):
        if not keyword:
            return "Keyword is required!",""
        try:        
            state_data_client = StateDataClient(self.connection_string, self.database)
            try:
                database_prompt, database_response = state_data_client.read_description_from_prompt(keyword)
                return "Response from Database", database_response
            except:
                database_prompt = None
                database_response = None
                pass 
        except Exception as err:
            return f"Error {err} in AskMeUIHandlers -> ask_chatgpt_database_response",""


    def ask_chatgpt_summarize(self, prompt):
        if not prompt:
            return "Prompt is required!",""
        try:        
            if self.api_key or self.azure_openai_key:
                operations = TextOperations()
                if self.azure_openai_deployment_name:
                    operations.set_azure_openai_api_key(self.azure_openai_key, self.azure_openai_api_base, self.azure_openai_deployment_name)
                    
                else:
                    operations.set_openai_api_key(self.api_key)
                    if self.model_name:
                        operations.set_model_name(self.model_name)                    
                    if self.org_id:
                        operations.set_org_id(self.org_id)
                    
                    
                response_message, response = operations.summarize(prompt)
                return response_message, response
            else:
                return self.NO_API_KEY_ERROR, ""
        except Exception as err:
            return f"Error {err} in AskMeUIHandlers -> ask_chatgpt_summarize",""
    

        
    def celeb_upload_save_real_generated_image(self, name, prompt, description, folder_name, real_picture, generated_picture):
        celeb_client = CelebDataClient(self.connection_string, self.database)
        cloudinary_client = CloudinaryClient(self.cloudinary_cloud_name, self.cloudinary_api_key, self.cloudinary_api_secret)
        # reading existing details real and generated image URLs, rest ignored
        try:
            l_name, l_prompt, l_description, real_picture_url, generated_image_url = celeb_client.get_celebs_response(name)
        except:
            l_name = None
            l_prompt = None
            l_description = ""
            real_picture_url = ""
            generated_image_url = ""
            pass
              
        if real_picture_url is not None and len(real_picture_url)==0:
            real_picture_url = ""
        if generated_image_url is not None and len(generated_image_url)==0:
            generated_image_url = ""

        # uploading real picture
        if real_picture is not None and real_picture_url is not None and len(real_picture_url)==0:
            cloudinary_client.set_folder_name(folder_name)
            real_picture_url = cloudinary_client.upload_image(real_picture, name)
        # uploading generated picture
        if generated_picture is not None and generated_image_url is not None and len(generated_image_url)==0:
            cloudinary_client.set_folder_name("Generated")
            generated_image_url = cloudinary_client.upload_image(generated_picture, name)

        # saving record back
        celeb_client.update_describe(name, prompt, description, real_picture_url, generated_image_url)
        return "Uploaded and saved real and generated images", name, prompt, description, real_picture_url, generated_image_url

    def update_description(self, name, prompt, description):
        celeb_client = CelebDataClient(self.connection_string, self.database)
        try:
            l_name, l_prompt, response, real_picture_url, generated_image_url = celeb_client.get_celebs_response(name)
        except:
            l_name = None
            l_prompt = None
            response = None
            real_picture_url = ""
            generated_image_url = ""
            pass
        try:
            celeb_client.update_describe(name, prompt, description, real_picture_url, generated_image_url)
        except Exception as err:
            print(f"Error {err} in AskMeUIHandlers -> update_description")    
    
    def describe_handler(self, name, prompt, folder_name, description, input_celeb_real_picture, input_celeb_generated_picture):
        name = name.strip()
        if not self.api_key or not prompt or not name :            
            return f"Name or prompt is not entered or {self.NO_API_KEY_ERROR}", "", "", "", None, None
        try:
            celeb_client = CelebDataClient(self.connection_string, self.database)
            try:
                l_name, l_prompt, l_description, real_picture_url, generated_image_url = celeb_client.get_celebs_response(name)
            except:
                l_name = None
                l_prompt = None
                l_description = ""
                real_picture_url = ""
                generated_image_url = ""
                pass

                
            if len(l_description)==0:
                if self.api_key or self.azure_openai_key:
                    operations = TextOperations()
                    if self.azure_openai_deployment_name:
                        operations.set_azure_openai_api_key(self.azure_openai_key, self.azure_openai_api_base, self.azure_openai_deployment_name)                        
                    else:
                        operations.set_openai_api_key(self.api_key)                        
                        operations.set_model_name(self.model_name)
                        if self.org_id:
                            operations.set_org_id(self.org_id)
               
                response_message, response = operations.chat_completion(prompt)
                description = response
            else:
                description = l_description
            
            cloudinary_client = CloudinaryClient(self.cloudinary_cloud_name, self.cloudinary_api_key, self.cloudinary_api_secret)
            if len(real_picture_url)==0 and input_celeb_real_picture is not None:
                cloudinary_client.set_folder_name(folder_name)
                real_picture_url = cloudinary_client.upload_image(input_celeb_real_picture, name)
            elif real_picture_url is None:
                real_picture_url = ""
            if len(generated_image_url)==0 and input_celeb_generated_picture is not None:
                cloudinary_client.set_folder_name("Generated")
                generated_image_url = cloudinary_client.upload_image(input_celeb_generated_picture, name)
            elif generated_image_url is None:
                generated_image_url = ""

            
            celeb_client.update_describe(name, prompt, description, real_picture_url, generated_image_url)
            return f"{self.LABEL_GPT_CELEB_SCREEN} - uploaded and saved", name, prompt, description, self.image_utils.url_to_image(real_picture_url), self.image_utils.url_to_image(generated_image_url)
                
            
        except Exception as err:
            return f"Error {err} in AskMeUIHandlers -> describe_handler", "", "", "", None, None
    
    def save_pdf_kb_searchData(self, keyword: str, title: str, url: str, summary: str):
        try: 
            kb_data_client = KBDataClient(self.connection_string, self.database)
            kb_data_client.save_kb_searchdata("pdf", keyword, title, url, summary)
        except Exception as err:
            return print(f"Error {err} in AskMeUIHandlers -> save_pdf_kb_searchData")

    def save_youtube_kb_searchData(self, output):
        try: 
            kb_data_client = KBDataClient(self.connection_string, self.database)
            kb_data_client.save_kb_searchdata(output)   
        except Exception as err:
            return print(f"Error {err} in AskMeUIHandlers -> save_youtube_kb_searchData")
        