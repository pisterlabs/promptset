from openai import OpenAI
import base64
import os
from tqdm import tqdm

class ChatgptVisionApi():
    
    def __init__(self,client):
        self.client=client

    def input_to_ai(self,image_name,prompt): #convert image to base64 format
        def convert_image_to_base64(image):
            with open(image, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read())
                return base64_image.decode("utf-8")
            
        base64_data = convert_image_to_base64(image_name)

        response = self.client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", 
                "text": prompt},
                {
                "type": "image_url",
                "image_url": 
                {
                    "url": f"data:image/jpeg;base64,{base64_data}",
                    'detail':'low' #another option high
                },
                },

            ],
            }
        ],
        max_tokens=300,
        )
        return str(response.choices[0]).split('content=')[1].split('role=')[0].replace('\\','')
    
    
    def save_caption(self,text,output_dir,image_name):
        f=open(output_dir+str(image_name).replace('.jpg','.txt'),'w')
        f.write(text)
        f.close()

def main():
    api='Enter API Key'
    client = OpenAI(api_key=api)

    input_dir='Input Image directory'
    output_dir='Output Image directory'

    prompt='Your Prompt'

    gpt=ChatgptVisionApi(client)
    already_done=[]
    
    for filename in (os.listdir(input_dir)): ## This is to start images whos caption are not generated
        if '.txt' in filename: 
            imgname=filename.replace('.txt','.jpg')
            if '.jpg' in imgname:
                already_done.append(imgname)
    
    new_img_list = [item for item in os.listdir(input_dir) if item not in already_done] #this will give list of images who doesnot have caption
    
    for filename in tqdm(new_img_list,desc="Processing items", unit="item"):
       
            if '.jpg' in filename:

                try:
                    result=gpt.input_to_ai(input_dir+filename,prompt)
                    gpt.save_caption(result,output_dir,filename) #caption will be save as image_name.txt
                    pass
                except Exception as e:
                    print(f"Error: {e}")

            
if __name__=='__main__':
    main()

