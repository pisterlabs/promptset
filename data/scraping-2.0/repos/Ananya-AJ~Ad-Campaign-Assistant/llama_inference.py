from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import yaml
import openai
import os
import replicate
import gradio as gr
from PIL import Image
from io import BytesIO
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# os.environ["OPENAI_API_KEY"] = "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"

def load_config():
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    os.environ['REPLICATE_API_TOKEN'] = cfg["api_keys"]["replicate_token"]
    openai_api_key = cfg["api_keys"]["openai_api_key"]
    sd_model = cfg["genai_model"]["stable_diffusion"]
    n_predictions = cfg["app_config"]["n_predictions"]
    llama_path = cfg["genai_model"]["llama_path"]
    return openai_api_key, llama_path,sd_model,n_predictions


def get_images(image_urls):
    img=[]
    for url in image_urls:
        res = requests.get(url,verify=False)
        if res.status_code == 200:
            image_bytes = BytesIO(res.content)
            image_bytes.seek(0)
            pic = Image.open(image_bytes)
            img.append(pic)
        else:
            print("Image not found")
    return img
 

def invoke_inference(prompt):    
    openai.api_key,local_path,sd_model,n_predictions = load_config()
   
    try:
        response = openai.Completion.create(
            
            model="text-davinci-003", 
            prompt=f"Summarize the following text:\n\n{prompt}",
            max_tokens=150 
        )
        data =  response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        
    # ("Models/llama-2-7b-32k-instruct.Q4_0.gguf")

    callbacks = [StreamingStdOutCallbackHandler()]
    template = """
Please act as a creative ad writer. You need to write a highly creative and engaging advertisement for the product described in the Ad_prompt. Focus on its key features

The ad should be in a format described in the prompt, weaving a compelling story around the product's features. Emphasize how the product can help buyers. Emphasize how it is important during the season or event described in the prompt.The ad should be festive and appealing to the right audience. 

Make sure the ad is not just informative but also evokes emotions and encourages immediate action. It should be at least 20 lines long, creative, and not stop abruptly. 

Ad_prompt: {data}
"""
    prompt = PromptTemplate(template=template, input_variables=["data"])
    # llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
    llm = GPT4All(model=local_path, backend="gptj",max_tokens = 1024, n_predict=256, callbacks=callbacks, verbose=True)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    ad_text = llm_chain.run(data)
   
    images = []
    try:
        response = replicate.run(sd_model,input={"prompt":data,"num_outputs":n_predictions})
        try:
            if response:
                image_urls = [url for url in response]
            if image_urls:
                images = get_images(image_urls)
            if images:
                return ad_text,images
        except Exception as e:
            raise gr.Error("Images could not be found",str(e))
    except Exception as e:
        raise gr.Error("replicate error",str(e))


