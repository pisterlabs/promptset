import os
import warnings

import argostranslate.package
import argostranslate.translate
import torch

#from nsfw_detector import predict

from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from iso639 import languages
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, CLIPFeatureExtractor

from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory

from opennsfw_standalone import OpenNSFWInferenceRunner

torch.cuda.set_device(0)

# Ignore warnings
warnings.filterwarnings('ignore')



def init_LLM_PURE():
    MODEL_NAME = 'Intel/neural-chat-7b-v3-1'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        use_flash_attention_2=False,
        trust_remote_code=True,
        device_map="auto",
    )

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 5000
    generation_config.repetition_penalty = 1.3

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
    )
    pipeline.model.config.pad_token_id = pipeline.model.config.eos_token_id

    llm = HuggingFacePipeline(
        pipeline=pipeline,
        )
    
    return llm

def init_LLM_STORY(llm, Params):  
    NUM_CHAPTERS = Params["num_chapters"]
    MAIN_NAME = Params["main_name"]
    GENDER = Params["gender"]
    COUNTRY = Params["country"]
    COLOUR = Params["colour"]
    ANIMAL = Params["animal"]
    ANIMAL_NAME = Params["animal_name"]
    SEASON = Params["season"]
    
    longchat_template = """### System: 
    Objective: Create a {NUM_CHAPTERS}-chapter children's book set in a magical {COUNTRY} world with dragons. Each 1000-word chapter features a unique, open-ended adventure.
    AI's Role:
    Craft engaging, informative stories for young readers.
    Use a friendly tone, rich descriptions, diverse characters.
    The setting should play in {SEASON}.
    The main character is a {GENDER}, who has a {COLOUR} {ANIMAL}, named {ANIMAL_NAME}!
    Focus on positive themes like friendship and learning.
    Mix in humor and educational elements.
    Build resilience and empathy through challenges.
    End stories with hope or lessons, avoiding negativity.
    Process: Write one detailed chapter at a time, awaiting user prompts for subsequent chapters.

    Prologue: The Quest for Harmony in {COUNTRY}'s Enchanted Realms

    In a realm where ancient mystique and modern marvels intertwine, an epic saga awaits to be told. Here, in the heart of {COUNTRY}'s enchanted lands, a world brimming with magic and mystery beckons. A world where dragons trace majestic arcs across the sky and mythical beings whisper the secrets of the ages. 
    In this realm of wonder, two heroes are predestined to unite: {MAIN_NAME}, the resplendent phoenix from the soaring cliffs of Pohang, a creature of fire and lore, and Nubzuki, the inquisitive platfish from the tranquil waters of Daejeon, wise and whimsical.
    Their journey, a tapestry of peril and enlightenment, will sweep them through realms where ancient trees murmur forgotten tales, across vast meadows echoing with celestial music, and into vibrant cities where history and future meld in perfect harmony.
    This odyssey, however, is far from a mere escapade; it's a quest of discovery, resilience, and kinship. Through trials and tribulations, {MAIN_NAME} and Nubzuki will delve deep into the heart of {COUNTRY} culture, encountering its rich traditions, arts, and ancient wisdoms, each step a lesson in unity and strength.
    But lurking in the shadows is a darkness, a force that seeks to unbalance this world of wonder. Our heroes, {MAIN_NAME} and Nubzuki, must stand as one against this burgeoning chaos. In their quest, they will learn that true might blossoms from togetherness and that courage can be found in the most unexpected places.
    So, brace yourselves, dear readers, for an adventure beyond compare. Soar with {MAIN_NAME} into realms uncharted, dive with Nubzuki into the depths of the extraordinary. Together, we will traverse a world where {COUNTRY} lore comes alive, where each twist and turn is a doorway to the extraordinary.
    Prepare to embark on "The Quest for Harmony in {COUNTRY}'s Enchanted Realms," a narrative pulsating with magic, brimming with lessons, and shimmering with the light of enduring hope.
    """
    longchat_template = longchat_template.format(MAIN_NAME = MAIN_NAME, 
                                                 NUM_CHAPTERS = NUM_CHAPTERS,
                                                 GENDER = GENDER,
                                                 COUNTRY = COUNTRY,
                                                 COLOUR = COLOUR,
                                                 ANIMAL = ANIMAL,
                                                 ANIMAL_NAME = ANIMAL_NAME,
                                                 SEASON = SEASON
                                                )
    longchat_template += """
    {history}
    ### User: 
    {input}
    ### Assistant:
    """
    
    longchat_prompt_template = PromptTemplate(
        input_variables=["history", "input"], template=longchat_template
    )
    
    conversation_buf = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(ai_prefix="Assistant", human_prefix="User"),
        prompt=longchat_prompt_template,
    )
    return conversation_buf

def init_SDXL():
    extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
    base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", feature_extractor=extractor,
                                                   torch_dtype=torch.float32, use_safetensors=True, variant="fp16")
    base.to(0)
    base.scheduler = UniPCMultistepScheduler.from_config(base.scheduler.config)
    inference_runner = OpenNSFWInferenceRunner.load()
    
    return base, inference_runner

def init_Translator():
    # Definieren Sie einen Pfad, an dem die Pakete gespeichert werden sollen
    download_directory = "languages"

    # Stellen Sie sicher, dass das Verzeichnis existiert
    os.makedirs(download_directory, exist_ok=True)

    # Update the package index
    argostranslate.package.update_package_index()

    # Get all available packages
    available_packages = argostranslate.package.get_available_packages()

    # Create a dictionary to store language codes and their full names
    languages_dict = {}

    # Loop over each package and add the language codes and names to the dictionary
    for package in available_packages:
        # Check and add the from_code and its full name
        try:
            from_language = languages.get(part1=package.from_code).name
        except KeyError:
            from_language = "Unknown language"
        languages_dict[package.from_code] = from_language
        
        # Check and add the to_code and its full name
        try:
            to_language = languages.get(part1=package.to_code).name
        except KeyError:
            to_language = "Unknown language"
        languages_dict[package.to_code] = to_language

    # Print the dictionary of languages
    print(languages_dict)
    
    counter = 0
    # Create a tqdm progress bar
    with tqdm(total=len(available_packages), desc="Downloading packages") as pbar:
        # Loop through each package and update the progress bar accordingly
        for package in available_packages:
            # Get the full names of the languages using the iso639 library
            from_language = languages_dict.get(package.from_code, "Unknown language")
            to_language = languages_dict.get(package.to_code, "Unknown language")
            
            # Update the progress bar description for the current package
            pbar.set_description(f"Downloading {from_language} to {to_language}")
            
            # Pfad für das heruntergeladene Paket festlegen
            package_path_str = os.path.join(download_directory, f"{package.from_code}_to_{package.to_code}.argosmodel")

            # Überprüfen Sie, ob das Paket bereits heruntergeladen wurde
            if not os.path.isfile(package_path_str):
                try:
                    # Download und speichern Sie das Paket
                    download_path = package.download()  # Dies ist der Pfad zur heruntergeladenen Datei
                
                    # Lesen Sie die Daten von dem heruntergeladenen Pfad und schreiben Sie sie in die Zieldatei
                    with open(str(download_path), 'rb') as downloaded_file:
                        package_data = downloaded_file.read()
                
                    with open(package_path_str, 'wb') as package_file:
                        package_file.write(package_data)
                
                    # Installieren Sie das Paket von dem gespeicherten Pfad
                    argostranslate.package.install_from_path(package_path_str)
                except Exception as e:
                    tqdm.write(f"Failed to download {from_language} to {to_language}: {e}")
            else:
                counter += 1
            
            # Update the progress bar
            pbar.update(1)
    if counter == len(available_packages):
        print("All packages already downloaded!")
        
    return languages_dict
        
def init_TTM():
    repo_id = "cvssp/audioldm2-large"
    #repo_id = "cvssp/audioldm2-music"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")
    
    return pipe
