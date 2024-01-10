import torch
import openai
import abc
import ast
import torch.nn.functional as F
import requests
import re

from config import settings as config
from collections import Counter
from itertools import chain
from PIL import Image
from io import BytesIO

#from models.LLaVA.llava.eval import run_llava
# from lavis.models import load_model_and_preprocess
# ------- LLAVA -------- is this correct?
from models.LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from models.LLaVA.llava.conversation import conv_templates, SeparatorStyle
from models.LLaVA.llava.model.builder import load_pretrained_model
from models.LLaVA.llava.utils import disable_torch_init
from models.LLaVA.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

with open('api.key') as f:
    openai.api_key = f.read().strip()
with open('api_org.key') as f:
    openai.organization = f.read().strip()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================== Base abstract model ========================== #
class BaseModel(abc.ABC):
    to_batch = False
    seconds_collect_data = 1.5  # Window of seconds to group inputs, if to_batch is True
    max_batch_size = 10  # Maximum batch size, if to_batch is True. Maximum allowed by OpenAI
    requires_gpu = True

    def __init__(self, gpu_number):
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        If to_batch is True, every arg and kwarg will be a list of inputs, and the output should be a list of outputs.
        The way it is implemented in the background, if inputs with defaults are not specified, they will take the
        default value, but still be given as a list to the forward method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """The name of the model has to be given by the subclass"""
        pass

    @classmethod
    def list_processes(cls):
        """
        A single model can be run in multiple processes, for example if there are different tasks to be done with it.
        If multiple processes are used, override this method to return a list of strings.
        Remember the @classmethod decorator.
        If we specify a list of processes, the self.forward() method has to have a "process_name" parameter that gets
        automatically passed in.
        See GPT3Model for an example.
        """
        return [cls.name]

# ========================== Specific Models ========================== #

class BLIPModel(BaseModel):
    """Model implementation for BLIP-2."""
    name = 'blip-2'
    to_batch = True
    max_batch_size = 32
    seconds_collect_data = 0.2
    # TODO: Create config YAML file
    def __init__(self, gpu_number=0, half_precision=False, 
                 blip_v2_model_type='blip2-flan-t5-xl'):
        super().__init__(gpu_number)

        assert blip_v2_model_type in ['blip2-flan-t5-xxl', 'blip2-flan-t5-xl', 'blip2-opt-2.7b', 'blip2-opt-6.7b',
                                      'blip2-opt-2.7b-coco', 'blip2-flan-t5-xl-coco', 'blip2-opt-6.7b-coco']
        
        #from lavis.models import load_model_and_preprocess
        
        """Imports a processor and BLIP-2 model from HuggingFace. 
        A Blip2Processor prepares images for the model and decodes the predicted tokens ID's back to text.
        """
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        

        with torch.cuda.device(self.dev):
            max_memory = {gpu_number: torch.cuda.mem_get_info(self.dev)[0]}
            self.processor = Blip2Processor.from_pretrained(f"Salesforce/{blip_v2_model_type}")
            try:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    f"Salesforce/{blip_v2_model_type}", load_in_8bit=half_precision,
                    torch_dtype=torch.float16 if half_precision else "auto",
                    device_map="sequential", max_memory=max_memory
                )
            except Exception as e:
                if "had weights offloaded to the disk" in e.args[0]:
                    extra_text = ' You may want to consider setting half_precision to True.' if half_precision else ''
                    raise MemoryError(f"Not enough GPU memory in GPU {self.dev} to load the model.{extra_text}")
                else:
                    raise e
        self.qa_prompt = "Question: {} Short answer:"
        self.caption_prompt = "a photo of"
        self.half_precision = half_precision
        self.max_words = 50
        
    @torch.no_grad()
    def caption(self, image, prompt=None):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt", padding="longest").to(self.dev)
        if self.half_precision:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=1000, min_length=1,
                                            do_sample=True, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text

    @torch.no_grad()
    def qa(self, image, question):
        inputs = self.processor(images=image, text=self.qa_prompt.format(question), return_tensors="pt", padding="longest").to(self.dev)
        if self.half_precision:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=1000, min_length=1,
                                            do_sample=True, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text

    def forward(self, image, questions=None, task='caption'):
        if not self.to_batch:
            image, questions, task = [image], [questions], [task]
        # if len(image) > 0 and 'float' in str(image[0].dtype) and image[0].max() <= 1:
        #    image = [im * 255 for im in image]
    
        # Separate into qa and caption batches.
        response = []
        if task == 'qa':
            prompts_qa = [self.qa_prompt.format(q) for q in questions]
            images_qa = [im for i, im in enumerate(image)]
        else:
            images_caption = [im for i, im in enumerate(image)]
        with torch.cuda.device(self.dev):
            response_qa = self.qa(images_qa, prompts_qa) if len(images_qa) > 0 else []
            response_caption = self.caption(images_caption) if len(images_caption) > 0 else []
        if task == 'qa':
            return response_qa
        else:
            return response_caption
        """
        images_qa, images_caption = [], []
        if task == 'qa':
            prompts_qa = [self.qa_prompt.format(self.pre_question(q)) for q, t in zip(question, task) if t == 'qa']
            images_qa = [im for i, im in enumerate(image) if task[i] == 'qa']
        else:
            images_caption = [im for i, im in enumerate(image) if task[i] == 'caption']
        with torch.cuda.device(self.dev):
            response_qa = self.qa(images_qa, prompts_qa) if len(images_qa) > 0 else []
            response_caption = self.caption(images_caption) if len(images_caption) > 0 else []
        response = []
        for t in task:
            if t == 'qa':
                response.append(response_qa.pop(0))
            else:
                response.append(response_caption.pop(0))
        if not self.to_batch:
            response = response[0]
        return response
        """

class SiglipModel(BaseModel):
    name = "siglip"
    to_batch = True
    max_batch_size = 32
    seconds_collect_data = 0.2
    """Model implementation for SIGLIP."""
    def __init__(self, gpu_number=0, siglip_model_type="ViT-SO400M-14-SigLIP-384"):
        super().__init__(gpu_number)
        with torch.cuda.device(self.dev):
            try:
                from open_clip import create_model_from_pretrained, get_tokenizer
                self.model, self.preprocess = create_model_from_pretrained(f"hf-hub:timm/{siglip_model_type}")
                self.model = self.model.to(self.dev)
                self.tokenizer = get_tokenizer(f"hf-hub:timm/{siglip_model_type}")
            except Exception as e:
                raise Exception(f"Could not load SIGLIP model: {e}")
    
    def prepare_images(self, images):
        image_stack = torch.stack([self.preprocess(image) for image in images])
        return image_stack
    
    def prepare_texts(self, texts):
        text_stack = self.tokenizer(texts, context_length=self.model.context_length)
        return text_stack

    @torch.no_grad()
    def forward(self, images, queries=NotImplementedError, top_k=1):
        if not self.to_batch:
            image, text = [image], [text]
        image_stack = self.prepare_images(images).to(self.dev)
        text_stack = self.prepare_texts(queries).to(self.dev)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image_stack)
            text_features = self.model.encode_text(text_stack)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            #print("Image features shape: ", image_features.shape, "Text features shape: ", text_features.shape)
            text_probs = torch.sigmoid(text_features @ image_features.T * self.model.logit_scale.exp() + self.model.logit_bias)
        # indices returns a matrix of shape [len(queries), top_k], where each row is the top_k indices for that text
        values, indices = torch.topk(text_probs, top_k)
        # TODO: implement functionality for multiple text prompts (batched)
        raw_images = []
        for i in range(len(queries)):
            #raw_images.append([indices[i][idx] for idx in range(3)])
            #indices = [indices[i][idx] for idx in range(top_k)] 
            raw_images.append([images[num] for num in [indices[i][idx].item() for idx in range(top_k)]])
        # TODO: also return the index
        return indices, raw_images

        
class GPTModel(BaseModel):
    name = 'gpt'
    to_batch = False
    requires_gpu = False

    def __init__(self, gpu_number=0, max_tries=1):
        super().__init__(gpu_number=gpu_number)
        # TODO: modify the prompting mechanism
        with open(config["gpt"]["qa_prompt"]) as f:
            self.qa_prompt = f.read().strip()
        self.temperature = config["gpt"]["temperature"]
        #self.n_votes = config["gpt"]["n_votes"]
        self.model = config["gpt"]["model"]
        self.max_tries = max_tries

    def call_llm(self, prompt, model, 
                 frequency_penalty=0, presence_penalty=0, 
                 max_tokens=1000, n=1, temperature=1):
        for _ in range(self.max_tries):
            try:
                completion = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Follow the directions given in the next prompt carefully."},
                    {"role": "user", "content": prompt}
                ],
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                n=n,
                temperature=temperature, 
                )
                output_message = completion.choices[0].message.content
                return output_message
            except Exception as e:
                print("Error: ", e)
                print("Trying again...")
                continue
        print("Could not get response from LLM.")
        return None
    

    @staticmethod
    def get_answer_helper(self, question, answer_choices, curr_frame, total_frames, caption, prev_info=None):
        with open('./prompts/base_prompt.txt') as f:
            prompt = f.read()
        prompt = prompt.replace('insert_question', question)
        prompt = prompt.replace('insert_choices', str(answer_choices))
        prompt = prompt.replace('insert_curr_frame', str(curr_frame))
        prompt = prompt.replace('insert_total_frames', str(total_frames))
        prompt = prompt.replace('insert_caption', caption[0])

        #print(prompt)
        output = self.call_llm(prompt, model=self.model)
        try:
            output_dict = ast.literal_eval(output)
            print("GETTING OUTPUT: ", output_dict)
            return output_dict
        except:
            print("ERROR: ", output)

    def final_select(self, question, choices, info):
        with open(config["gpt"]["final_select_prompt"]) as f:
            prompt = f.read()
        prompt = prompt.replace('insert_question', question)
        prompt = prompt.replace('insert_choices', str(choices))
        prompt = prompt.replace('insert_info', str(info))
        #print(prompt)
        output = self.call_llm(prompt, model=self.model)
        try:
            output_dict = ast.literal_eval(output)
            print("GETTING FINAL OUTPUT: ", output_dict)
            return output_dict
        except:
            print("ERROR: ", output)
            return output


    # initial cleaning for reference QA results
    @staticmethod
    def process_answer(answer):
        """Strips whitespace, periods, commas, and filler words"""
        answer = answer.lstrip()  # remove leading spaces (our addition)
        answer = answer.replace('.', '').replace(',', '').lower()
        to_be_removed = {'a', 'an', 'the', 'to', ''}
        answer_list = answer.split(' ')
        answer_list = [item for item in answer_list if item not in to_be_removed]
        return ' '.join(answer_list)

    @staticmethod
    def get_union(lists):
        return list(set(chain.from_iterable(lists)))

    def get_qa_fn(self, prompt):
        response = self.query_gpt3(prompt, model=self.model, max_tokens=5, logprobs=1, stream=False,
                                   stop=["\n", "<|endoftext|>"])
        return response

    def get_general(self, prompt):
        """Gets the general response from GPT-3."""
        response = self.call_llm(prompt, model = self.model)
        return response

    # TODO: implement generic forward functionality with case handling
    def forward(self, prompt, process_name=None):
        # temporary for now
        response = self.get_general(prompt)
        return response

class LLAVA(BaseModel):
    name = 'llava'
    to_batch = False
    requires_gpu = True

    def __init__(self, gpu_number=0, max_tries=1):
        super().__init__(gpu_number)
        with torch.cuda.device(self.dev):
            disable_torch_init()
            self.model_name = get_model_name_from_path(config["llava"]["model_path"])
            self.tokenizer, self.model, self.image_processor, self.devcontext_len = load_pretrained_model(
                config["llava"]["model_path"], config["llava"]["model_base"], self.model_name, 
                load_8bit=False, load_4bit=False,
                device_map="auto", device=self.dev
            )

    def image_parser(self, image_file):
        out = image_file.split(",")
        return out

    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out
    
    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image_file
        
    def forward(self, image_list, question):
        """Forward method
        Args:
            - images
            - question
        """
        qs = question
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if config["llava"]["conv_mode"] is not None and conv_mode != config["llava"]["conv_mode"]:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, config["llava"]["conv_mode"], config["llava"]["conv_mode"]
                )
            )
        else:
            config["llava"]["conv_mode"] = conv_mode

        conv = conv_templates[config["llava"]["conv_mode"]].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        
        #image_files = self.image_parser(image_list)
        #images = self.load_images(image_files)
        # TODO: Check if this works
        images = image_list

        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.dev, dtype=torch.float16)
    
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.dev)
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if config["llava"]["temperature"] > 0 else False,
                temperature= config["llava"]["temperature"],
                top_p=config["llava"]["top_p"],
                num_beams=config["llava"]["num_beams"],
                max_new_tokens=config["llava"]["max_new_tokens"],
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )


        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs