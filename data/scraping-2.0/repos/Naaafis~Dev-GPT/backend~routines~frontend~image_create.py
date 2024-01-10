from reactManager import ReactAppManager
import autogen
from autogen import *
import requests
import json
import pdb
import os
import re

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from autogen import AssistantAgent, Agent, UserProxyAgent, ConversableAgent

from autogen.img_utils import get_image_data, _to_pil
from termcolor import colored
import random

from openai import OpenAI
import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt

from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from diskcache import Cache

def dalle_call(client: OpenAI, model: str, prompt: str, size: str, quality: str, n: int) -> str:
    """
    Generate an image using OpenAI's DALL-E model and cache the result.

    This function takes a prompt and other parameters to generate an image using OpenAI's DALL-E model.
    It checks if the result is already cached; if so, it returns the cached image data. Otherwise,
    it calls the DALL-E API to generate the image, stores the result in the cache, and then returns it.

    Args:
        client (OpenAI): The OpenAI client instance for making API calls.
        model (str): The specific DALL-E model to use for image generation.
        prompt (str): The text prompt based on which the image is generated.
        size (str): The size specification of the image. TODO: This should allow specifying landscape, square, or portrait modes.
        quality (str): The quality setting for the image generation.
        n (int): The number of images to generate.

    Returns:
    str: The image data as a string, either retrieved from the cache or newly generated.

    Note:
    - The cache is stored in a directory named '.cache/'.
    - The function uses a tuple of (model, prompt, size, quality, n) as the key for caching.
    - The image data is obtained by making a secondary request to the URL provided by the DALL-E API response.
    """
    # Function implementation...
    cache = Cache('.cache/')  # Create a cache directory
    key = (model, prompt, size, quality, n)
    if key in cache:
        return cache[key]

    # If not in cache, compute and store the result
    response = client.images.generate(
          model=model,
          prompt=prompt,
          size=size,
          quality=quality,
          n=n,
        )
    image_url = response.data[0].url
    img_data = get_image_data(image_url)
    cache[key] = img_data

    return img_data


def extract_img(agent: Agent) -> PIL.Image:
    """
    Extracts an image from the last message of an agent and converts it to a PIL image.

    This function searches the last message sent by the given agent for an image tag,
    extracts the image data, and then converts this data into a PIL (Python Imaging Library) image object.

    Parameters:
        agent (Agent): An instance of an agent from which the last message will be retrieved.

    Returns:
        PIL.Image: A PIL image object created from the extracted image data.

    Note:
    - The function assumes that the last message contains an <img> tag with image data.
    - The image data is extracted using a regular expression that searches for <img> tags.
    - It's important that the agent's last message contains properly formatted image data for successful extraction.
    - The `_to_pil` function is used to convert the extracted image data into a PIL image.
    - If no <img> tag is found, or if the image data is not correctly formatted, the function may raise an error.
    """
    # Function implementation...
    img_data = re.findall("<img (.*)>", agent.last_message()["content"])[0]
    pil_img = _to_pil(img_data)
    return pil_img



class DALLEAgent(ConversableAgent):
    def __init__(self, name, llm_config: dict, **kwargs):
        super().__init__(name, llm_config=llm_config, **kwargs)
        
        try:
            config_list = llm_config["config_list"]
            api_key = '?'
        except Exception as e:
            print("Unable to fetch API Key, because", e)
            api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key='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')
        self.register_reply([Agent, None], DALLEAgent.generate_dalle_reply)
        
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        # override and always "silent" the send out message; 
        # otherwise, the print log would be super long!
        super().send(message, recipient, request_reply, silent=True)
        
    def generate_dalle_reply(self, messages: Optional[List[Dict]], sender: "Agent", config):
        """Generate a reply using OpenAI DALLE call."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        prompt = messages[-1]["content"]
        # TODO: integrate with autogen.oai. For instance, with caching for the API call
        img_data = dalle_call(
            client=self.client,
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024", # TODO: the size should be flexible, deciding landscape, square, or portrait mode.
            quality="standard",
            n=1,
        )
        out_message = f"<img {img_data}>"
        return True, out_message

    

class DalleCreator(AssistantAgent):
    def __init__(self, files, imgcreate_read_config, imgcreate_write_config, imgcreate_prompt_write_config, n_iters=2, **kwargs):
        """
        Initializes a DalleCreator instance.
        
        This agent facilitates the creation of visualizations through a collaborative effort among 
        its child agents: dalle and critics.
        
        Parameters:
            - n_iters (int, optional): The number of "improvement" iterations to run. Defaults to 2.
            - **kwargs: keyword arguments for the parent AssistantAgent.
        """
        super().__init__(**kwargs)
        self.file_path = files
        self.gpt4v_base_config = imgcreate_read_config
        self.dalle_base_config = imgcreate_write_config
        self.imgcreate_prompt_write_config = imgcreate_prompt_write_config
        self.register_reply([Agent, None],
                            reply_func=DalleCreator._reply_user,
                            position=0)
        self._n_iters = n_iters

    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        img_prompt = messages[-1]["content"]

        ## Define the agents
        self.critics = MultimodalConversableAgent(
            name="Reviewer",
            system_message=
            """
            As a reviewer, critically evaluate the image you see and improve on the prompt the dall-e-3 image creator used. IF theres feedback from the user after "FEEDBACK".
            Make sure the image is closely relatable with the content from file path with the feedback update in it. If the feeedback is not included, improve on the prompt 
            the dall-e-3 image creator used to make sure the user feedback is represented in new image generation.
                   
            If you think all the files are good to go, reply with 'done'. Tell the other agents to do the same.
            """,
            llm_config=self.gpt4v_base_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
        )

        self.dalle = DALLEAgent(name="Dalle", llm_config=self.dalle_base_config, max_consecutive_auto_reply=0)

        # Data flow begins
        self.send(message=img_prompt, recipient=self.dalle, request_reply=True)
        img = extract_img(self.dalle)
        #plt.imshow(img)
        #plt.axis('off')  # Turn off axis numbers
        #plt.show()
        print("Image PLOTTED")
        
    
        
        self.elements_inspector = MultimodalConversableAgent(
            name="Elements inspector",
            system_message=
            """
            As a web design elements inspector, your work is to carefully inspect the image of web design you see and list and describe web elements you see.
            When you describing the image, you would have access to the original prompt to generate the image, combine with it to come up with your evaluation,
            you always start with a general description of style of the whole design before going to every element you saw
            Those elements will each have a name for it, what it does, then you should describe its location on the image: (is it top right
            for user icon? is it bottom for footer), describe the color, style and fonts.
            
            Write your response after "ELEMENTS: "       
            If you think all the files are good to go, reply with 'done'. Tell the other agents to do the same.
            """,
            llm_config=self.gpt4v_base_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
        )
        
        
        for i in range(self._n_iters):
            # Downsample the image s.t. GPT-4V can take
            img = extract_img(self.dalle)
            img.save("./front_end/saves/design.png")
            #img.save(f"betterresult{i}.png")
            smaller_image = img.resize((128, 128), Image.Resampling.LANCZOS)
            #smaller_image.save(f"result{i}.png")
            

            self.msg_to_critics = f"""USER PROMPT: {img_prompt}.
            GENERATED IMAGE <img ./front_end/saves/design.png>.
            
            Follow your system instruction to do the work and give the result in the following guideline.
            
            Write them in two seperate paragraphs and make sure the critics is in the first paragraph after "CRITICS: "
            and the improved prompt in the second paragraph after the phrase "PROMPT: "
            """
            self.send(message=self.msg_to_critics,
                           recipient=self.critics,
                           request_reply=True)
            feedback = self._oai_messages[self.critics][-1]["content"]
            
            img_prompt = re.findall("PROMPT: (.*)", feedback)[0]
            self.send(
                message=img_prompt,
                recipient=self.dalle,
                request_reply=True)
            
            #plt.imshow(img)
            #plt.axis('off')  # Turn off axis numbers
            #plt.show()
            #print(f"Image {i} PLOTTED")
            
            
            self.msg_to_element = f"""USER PROMPT: {img_prompt}.
            GENERATED IMAGE <img ./front_end/saves/design.png>.
            
            Follow your system instruction to do the work and give the result in the following guideline.
            
            Make sure your response is after "ELEMENTS: "
            """
            self.send(message=self.msg_to_element,
                           recipient=self.elements_inspector,
                           request_reply=True)
            elements = self._oai_messages[self.elements_inspector][-1]["content"]
            #print("ELEMENTS: " + elements)
            element_file='./front_end/saves/web_elements.txt' 
            with open(element_file, 'w') as filetowrite:
                 filetowrite.write(elements)
            
            rework_file='./front_end/saves/user_prompt.txt' 
            with open(rework_file, 'w') as filetowrite:
               filetowrite.write(img_prompt)
            #self.send(message=img_prompt, recipient=self.prompt_updater, request_reply=True)

        feedback_reset = ""
        feedback_file='./front_end/saves/user_feedback.txt' 
        with open(feedback_file, 'w') as filetowrite:
            filetowrite.write(feedback_reset)
        img.save(f"./front_end/saves/design.png")
        img.save(f"./front_end/public/images/design.png")
        #self.react_manager.create_new_file("./saves/", "design.png", result1.png)
        return True, "design.png"
    

class ImageCreateRoutine:
    """
    Routine to create image with dalle 3
    This routine will need to read enhanced prompt from file and gpt4-vision for review
    """
    def __init__(self, base_config, imgcreate_prompt_write_config, imgcreate_read_config, imgcreate_write_config, img_create_function_map):
        self.imgcreate_prompt_write_config = imgcreate_prompt_write_config
        self.base_config = base_config
        self.gpt4v_base_config = imgcreate_read_config
        self.dalle_base_config = imgcreate_write_config
        self.imgcreate_writing_function_map = img_create_function_map
        
        termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

        IMAGECREATE_AUTO_REPLY = """
            Ensure that you have fully fullfilled the description in prompt and converted it into an image. Reflect on the current image generated.
            Are there any part of prompt that the dalle-e-3 that might need more detailed description or have been overlooked?
            Consider re-evaluating the enhanced prompt for completeness and accuracy. Remember there are a image-to-text model to turn
            this image and prompt to evaluate a code design. If you are satisfied with the prompt, please reply with 'done'.
        """
    

        self.client = UserProxyAgent(
            name="client",
            max_consecutive_auto_reply=5,
            function_map=img_create_function_map,
            human_input_mode="NEVER",
            default_auto_reply=IMAGECREATE_AUTO_REPLY,
            code_execution_config=False,
        )
            
        IMAGECREATOR_READING_AGENT_MESSAGE = """
            You are responsible for reading and analyzing the content of the file at the provided file path. Your goal is to fully retrieve information from the datapath. 
            Use the 'read_file' function, splitting 'file_path' into the directory and file name. For instance, if 'file_path' is 'src/utils/helper.js', 
            your read_file call should be with 'src/utils' as the directory and 'helper.js' as the file name. Look for areas that require further development or clarification.
            
            In the feedback.txt, those are user's requirement that unsatisfied with current image, you must notify the image creator and image reviewer about the changes
            the user wish to make. The lines later in the file means newer instructions. 

            
            Remember as a prompt reader, you are not responsible for generating any images or write anything. But because you know the other relevant files, you can provide guidance on prompt description
            To this end, you can consider reading the other files and provide guidance for the image creator and image reviewer. When you finish reading and retrieving the full content of the file. 
            Send the description to both the image creator and the image reviewer so the reviewer might later comment on the image generated by image reviewer.
        """
        #Combine with the prompt you read from user_prompt.txt, you are going to write combined feedback of the files you read.    
        
        self.prompt_reader = AssistantAgent(
            name="prompt_reader",
            llm_config=self.imgcreate_prompt_write_config,
            system_message=IMAGECREATOR_READING_AGENT_MESSAGE
        )
        
        IMAGECREATOR_AGENT_MESSAGE = """
            You are responsible for generating image according to the the prompt provided by the prompt reader. Your job is to take the prompt
            and turn them into image for web design AIs that create actual code later. You will be in communication with a UI/UX designer to further improve the image
            You only work with the provided file path but you gain insight from the prmompt_reader about the contents of the other relevant files.
            When you finish your work, using the 'create_new_file' function to save the image you generated into the same folder of the known filepath using appropriate file name related to the prompt
            When using the 'write_to_file' function, remember to split 'file_path' into the directory and file name. 
        """

        # self.image_creator = AssistantAgent(
        #     name="prompt_writer",
        #     llm_config=self.dalle_base_config,
        #     system_message=IMAGECREATOR_AGENT_MESSAGE
        # )
        
        self.image_creator = DALLEAgent(name="Dalle", llm_config=self.dalle_base_config)
        #self.image_creator = DALLEAgent(name="Dalle", llm_config=self.dalle_base_config)

        IMAGECREATOR_AGENT_SYSTEM_MESSAGE = """
            As a reviewer, critically evaluate the image you see and improve on the prompt the dall-e-3 image creator used.
            Make sure the image generated is closely relatable with the content from file path and identify different parts of the web design at locations in the image.
            Some of the part design might include: main banner, typography, navigation bar, footer, buttons. The design should be replicable
            in code designs by other agents.
            
            Use the 'read_file' function to access the user_prompt.txt for prompts, splitting 'file_path' as needed. 
            Provide feedback or suggest improvements to ensure high-quality image generated. Make sure that the image generated do not conflate and halluciate
            on the contents of the other relevant files. To this end, you may want to look into the contents of the other files. Provide suggestions to the image_creator
            to ensure that the stubs are accurate and relevant to the high-level task.
            
            If you think all the files are good to go, reply with 'done'. Tell the other agents to do the same.
        """

        self.image_reviewer = AssistantAgent(
            name="prompt_reviewer",
            llm_config=self.gpt4v_base_config,
            system_message=IMAGECREATOR_AGENT_SYSTEM_MESSAGE
        )
    
    def image_create(self, file_path,  user_prompt, user_feedback = ""):
        
        # self.image_create_groupchat = GroupChat(
        #    agents=[self.client, self.prompt_reader], messages=[], max_round=20
        # )
        
        # manager = GroupChatManager(groupchat=self.image_create_groupchat, llm_config=self.base_config)
        
        # IMAGECREATE_PROMPT = """
        #     Our high-level task, '{high_level_task}', involves working across multiple files included in the high level task description. 
        #     Currently, focus on generating the image from the prompt in './saves/user_prompt.txt'. 
        #     Create a new file saving the image with name using date and time for unique names. Generate the image in the folder of the './saves/'
        # """
        
        # self.client.initiate_chat(
        #     manager,
        #     message=IMAGECREATE_PROMPT.format(high_level_task=high_level_task)
        # )
        
        creator = DalleCreator(
            files = file_path,
            imgcreate_read_config = self.gpt4v_base_config,
            imgcreate_write_config = self.dalle_base_config,
            imgcreate_prompt_write_config = self.imgcreate_prompt_write_config,
            name="DALLE Creator!",
            max_consecutive_auto_reply=0,
            system_message="Help me coordinate generating image",
            llm_config = self.base_config,
        )
        
        user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            function_map=self.imgcreate_writing_function_map,
            max_consecutive_auto_reply=0
        )
        
        combined_message = "PROMPT: " + user_prompt + "FEEDBACK: " + user_feedback
        user_proxy.initiate_chat(creator, message = combined_message)
                             
        # Return success message or any relevant output
        return "Image successfully generated "
    
    
    