#first function 
def download_blip():
    import pip
    import git
    import sys
    import os
    git.Git("https://github.com/salesforce/BLIP").clone("https://github.com/salesforce/BLIP.git")
    sys.path.append('../BLIP/')
    os.chdir('./BLIP')
    print("Git Clone completed...")
#second function
def model_download():
    from PIL import Image
    import requests
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #import gradio as gr
    from models.blip import blip_decoder
    image_size = 384
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'    
    model = blip_decoder(pretrained=model_url, image_size=384, vit='large')
    model.eval()
    model = model.to(device)
    from models.blip_vqa import blip_vqa
    image_size_vq = 480
    transform_vq = transforms.Compose([
        transforms.Resize((image_size_vq,image_size_vq),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    model_url_vq = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'    
    model_vq = blip_vqa(pretrained=model_url_vq, image_size=480, vit='base')
    model_vq.eval()
    model_vq = model_vq.to(device)
    print('downlod model finished complited !!')
#third function
def image_caption(image_path):
    def load_demo_image(image_size,device,image_path):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        from PIL import Image
        
        #image_path = '2.png' 
        #raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB') 
        raw_image = Image.open(image_path).convert('RGB')
        w,h = raw_image.size
        #display(raw_image.resize((w//5,h//5)))        
        transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image).unsqueeze(0).to(device)   
        return image
    image_size = 384
    #filename='2.jpg'
    from models.blip import blip_decoder
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = load_demo_image(image_size=image_size, device=device,image_path=image_path)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'    
    model = blip_decoder(pretrained=model_url, image_size=384, vit='large')
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        return caption[0]

def english_story(prompt, API_KEY,engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=4000, freq_pen=0.0, pres_pen=0.5, stop=['<<END>>']):
    #API_KEY='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'
    import os
    import openai
    openai.api_key = os.getenv(API_KEY)
    openai.api_key = API_KEY
   # translator = Translator()
    #prompt = translator.translate(prompt, dest='en')
    max_retry = 1
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                model=engine,
                prompt='write a  short story about '+ prompt,
                temperature=temp,
                max_tokens=tokens,
                #top_p=top_p,
               # frequency_penalty=freq_pen,
               # presence_penalty=pres_pen,
               # stop=[" User:", " AI:"]
                )
            text = response['choices'][0]['text'].strip()
            #print(text)
         #   filename = '%s_gpt3.txt' % time()
          #  with open('gpt3_logs/%s' % filename, 'w') as outfile:
              # outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            #sleep(1)
def arabic_story(prompt, API_KEY,engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=4000, freq_pen=0.0, pres_pen=0.5, stop=['<<END>>']):
    #API_KEY='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'
    from googletrans import Translator
    import os
    import openai
    openai.api_key = os.getenv(API_KEY)
    openai.api_key = API_KEY
    translator = Translator()
    #prompt = translator.translate(prompt, dest='en')
    max_retry = 1
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                model=engine,
                prompt='write a  short story about '+ prompt,
                temperature=temp,
                max_tokens=tokens,
                #top_p=top_p,
               # frequency_penalty=freq_pen,
               # presence_penalty=pres_pen,
               # stop=[" User:", " AI:"]
                )
            text = response['choices'][0]['text'].strip()
            result=translator.translate(text,dest='ar')
            text=result.text
            
            #print(text)
         #   filename = '%s_gpt3.txt' % time()
          #  with open('gpt3_logs/%s' % filename, 'w') as outfile:
              # outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            #sleep(1)
#upload image
image_path='car.jpg'
from IPython.display import Image
Image(image_path, width = 600, height = 300)
caption=image_caption(image_path)
caption