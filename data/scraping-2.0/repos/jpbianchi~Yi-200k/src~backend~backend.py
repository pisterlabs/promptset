import modal
from modal import web_endpoint
import os
from typing import Dict

stub = modal.Stub("GPU_server")
llms = modal.Image.debian_slim().pip_install(
    "transformers==4.35.2", 
    "accelerate==0.24.1",
    "SentencePiece==0.1.99",
    )

@stub.function(image=llms, 
               gpu=modal.gpu.A100(memory=80, count=1), # TAKES A LOT OF TIME TO DOWNLOAD... 
               timeout=1200,  # <<< we need more for the 34B model
               )
def llm_prompt_A100(model="01-ai/Yi-34B",
               prompt="What is your name?"):

    if model == "01-ai/Yi-34B":
        pass

@stub.function(image=llms, 
               gpu=modal.gpu.A100(count=1), 
               timeout=600,
               # secrets are available in the environment with os.environ["SECRET_NAME"]
               secret=modal.Secret.from_name("my-huggingface-secret"))
@web_endpoint(method="POST")
def llm_prompt(item: Dict):
    model, prompt = item['model'], item['prompt']
    # return f"Prompt received:{prompt} and model:{model} of type {type(model)}"

    if True or model in ("01-ai/Yi-6B", "01-ai/Yi-34B", "TheBloke/Yi-34B-200K-AWQ"):

        # model_name = "01-ai/Yi-34B-Chat-4bits"
                
        from transformers import AutoModelForCausalLM, AutoTokenizer

        messages = [
                {"role": "user", 
                "content": prompt
                }
            ]

        tokenizer = AutoTokenizer.from_pretrained(model, 
                                                  use_fast=False)
        
        input_ids = tokenizer.apply_chat_template(conversation=messages, 
                                                  tokenize=True, 
                                                  add_generation_prompt=True, 
                                                  return_tensors='pt')
 
        model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                torch_dtype='auto'
        ).eval()
        
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], 
                                    skip_special_tokens=True)

        # Model response: "Hello! How can I assist you today?"
        print(response)
        
        
        return f"Prompt received:{prompt} and model:{model}"
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # import transformers
        # import torch

        # tokenizer = AutoTokenizer.from_pretrained(model)
        # print("Tokenizer created!")
        # pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     torch_dtype=torch.bfloat16,
        #     trust_remote_code=True,
        #     device_map="auto",
        # )
        # # print("Pipeline created!")
        
        # responses={}
        
        # for animal in animals:

        #     sequences = pipeline(
        #         prompt.format(animal=animal),
        #         max_length=200,
        #         do_sample=True,
        #         top_k=1,
        #         num_return_sequences=1,
        #         eos_token_id=tokenizer.eos_token_id,
        #     )
        #     print(f"Sequence created for {animal}")
        #     for seq in sequences:
        #         print(f"Result: {seq['generated_text']}")
            
        #     # let's remove the prompt that gets repeated in the response
        #     response = sequences[0]['generated_text'].split('Features:')[1:]

        #     print("Response is found")
        #     print(f'Features of {animal} are:', response)
        #     responses[animal] = response
            
        # return responses
        
    else:
        pass
        # trying to run falcon40b with quantization
        # from langchain import PromptTemplate
        # from langchain.chains import LLMChain
        # from langchain.llms import HuggingFaceHub
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        #     import BitsAndBytesConfig
        #     import transformers
        #     import torch
        #     # Configuring the BitsAndBytes quantization
        #     bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     device_map = 'auto',
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        #     )
        #     # Loading the Falcon model with quantization configuration
        #     model = FalconForCausalLM.from_pretrained(
        #     modelname,
        #     quantization_config=bnb_config,
        #     )
        #     tokenizer = AutoTokenizer.from_pretrained(model)
        #     # Set the padding token to be the same as the end-of-sequence token
        #     # tokenizer.pad_token = tokenizer.eos_token

        #     pipeline = transformers.pipeline(
        #         "text-generation",
        #         model=model,
        #         tokenizer=tokenizer,
        #         torch_dtype=torch.bfloat16,
        #         trust_remote_code=True,
        #         device_map="auto",
        #     )

        #     sequences = pipeline(
        #     """Marc is an expert in human and animal physiology and also a Machine Learning engineer specialised in computer vision.
        #     Please provide all the features of a cat that allows to recognize it for sure.
        #     Be exhaustive.  List all body parts.
        #     \nMe: Hello, Marc!\nMarc:""",
        #     max_length=200,
        #     do_sample=True,
        #     top_k=10,
        #     num_return_sequences=1,
        #     eos_token_id=tokenizer.eos_token_id,
        # )
        #     for seq in sequences:
        #         print(f"Result: {seq['generated_text']}")
        #    return response

# @stub.local_entrypoint()
# def test_method(prompt="This is test_method", model="tiiuae/falcon-7b-instruct"):
#     output = model
#     return output
  
  # deploy it with
  # modal token set --token-id ak-xxxxxx --token-secret as-xxxxx # given when we create a new token
  # modal deploy backend.py
  # View Deployment: https://modal.com/apps/jpbianchi/GPU_server <<< use this project name