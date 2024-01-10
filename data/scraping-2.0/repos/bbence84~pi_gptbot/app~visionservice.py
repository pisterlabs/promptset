import cv2
import os
import time
import openai
import base64

from dotenv import load_dotenv
load_dotenv()


class VisionService:
    def __init__(self, default_language="Hungarian"):
        api_type = os.getenv('OPENAI_API_TYPE')
        if api_type == 'azure':
            self.client = openai.AzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_GPT4_API_KEY"),
                azure_endpoint = os.getenv('AZURE_OPENAI_GPT4_ENDPOINT'),
                api_version = os.getenv('AZURE_OPENAI_GPT4_VERSION'),                          
            )                    
        else:
            self.client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )   

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

        
    def get_whats_visible_on_camera(self):
        file_path = os.path.abspath(os.path.dirname(__file__)) + "/"
        file_name = 'capture.png'      
        local_file = os.path.join(file_path, file_name)
        cam = cv2.VideoCapture(0)
        result, image = cam.read()
        rotated=cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if result:
            success = cv2.imwrite(local_file, rotated)
            if success == False:
                print("Error saving image")
                return 'Nothing'
        cam.release()
        base64_image = self.encode_image(local_file)     

        response = self.client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_GPT4_DEPLOYMENT'), 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 
                        '''Act as a robot who has vision. Describe what you can see. Respond in Hungarian.'''},
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }                              
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def checkcamera(self):
        print('Checking Device...... \n')
        ret = os.popen("ls /dev/video*").read()
        video_list = []
        for i in ret.split('\n'):
            video_list.append((i.split('/')[-1][5:]))
        for i in video_list[:-1]:
            if int(i) < 10:
                print('Current device: video' + i + '\n')
                return int(i)
        return None    
            
if __name__ == "__main__":
    vision_service = VisionService()
    print(vision_service.get_whats_visible_on_camera())