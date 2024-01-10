import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import streamlit as st
import re
import ast
import torch
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.llms import HuggingFacePipeline, LlamaCpp
import gc
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

album = []
def generate(prompt, pipeline):
    with torch.inference_mode():
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        image = pipeline(prompt=prompt, num_inference_steps=20, guidance_scale=8.0, height=512, width=768).images[0]
    return image

def string_to_dict(input_string):
    scenes = {}
    scene_lines = input_string.split('\n')

    for line in scene_lines:
        line = line.strip()
        if line:
            scene_parts = line.split(':', 1)
            if len(scene_parts) == 2:
                scene_num = scene_parts[0].strip()
                description = scene_parts[1].strip()
                scenes[scene_num] = description
    return scenes

def string_to_list(input_string):

    # Find the list using regular expression
    match = re.search(r'\[.*?\]', input_string, re.DOTALL)

    if match:
        extracted_list_str = match.group(0)
    
        # Convert the extracted string to a list using ast.literal_eval
        extracted_list = ast.literal_eval(extracted_list_str)
    
        return extracted_list
    else:
        print("No list found in the input string.")

def llm_create_story(llm, suggestion):
    template = """
    The context of your movie is {suggestion}
    Return a list of important events in your movie the format of a Python list like this ["Detailed description of Event 1", "Detailed description of Event 2", "Detailed description of Event 3", "Detailed description of Event 4"]
    Only return your Python list of important events, do not add any unnecessary information of your movie
     """

    prompt = PromptTemplate(input_variables=['suggestion'], template=template)
    with torch.inference_mode():
        llm_chain = LLMChain(
            prompt=prompt,
            llm=llm
        )
        story = llm_chain.run(suggestion)
    return story

def create_image_album(input_string, pipeline):
    scenes = string_to_dict(input_string)
    for scene_num, description in scenes.items():
        image_filename = f"{scene_num}.png"
        img = generate(f"{description}, high resolution, realistic", pipeline)
        print(f"{image_filename} finished")
        st.image(img, use_column_width=True, caption=description)
        album.append(img)

def create_image_album_llm(input_string, pipeline):
    scenes = string_to_list(input_string)
    for i in range(len(scenes)):
        image_filename = f"{i}.png"
        img = generate(f"{scenes[i]}, high resolution, realistic", pipeline)
        print(f"{image_filename} finished")
        st.image(img, use_column_width=True, caption=scenes[i])
        album.append(img)
        
@st.cache_resource
def load_diffuser_model(device):
    model_id = "prompthero/openjourney"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

@st.cache_resource
def load_llm_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
  

    if model_basename is not None:
        if ".ggml" in model_basename:
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm
    
def print_cuda_memory_usage():
    allocated_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_cached()
    
    print(f"CUDA Memory Allocated: {allocated_memory / 1024 ** 3:.2f} GB")
    print(f"CUDA Memory Cached: {cached_memory / 1024 ** 3:.2f} GB")

def clear_cuda_memory():
    torch.cuda.empty_cache()
    st.cache_resource.clear()
    torch.cuda.empty_cache()
    gc.collect()
        

    