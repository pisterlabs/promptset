from generationFunctions import GenerationFunctions
from animdiffwrapper import generateGif
from load_llama_model import getllama
import builtins
import contextlib
from text_to_phonemes import processAudio



import sys
sys.path.append('.\AAAI22-one-shot-talking-face')


from test_script import test_with_input_audio_and_image, parse_phoneme_file
from exampleScenes import exampleScenesPrompt, exampleScenesResult
from exampleChapters import examplechapterPrompt, exampleChapterResults
from example_screenplay import exampleScreenplayPrompt, exampleScreenplayResult
import datetime
import uuid
import logging
# from riffusion import get_music
# import riffusion
from worldObject import WorldObject, ListObject
from templates import templates
from mubert import generate_track_by_prompt
import IPython.display as ipd
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub

import pits.app as pits

import traceback


from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler, DiffusionPipeline
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny, StableDiffusionXLImg2ImgPipeline
import time
from torch import autocast
import ipywidgets as widgets
from ipywidgets import Audio  # no good, doesn't stop when clear display
import numpy
import numpy as np
from io import BytesIO
from pydub import AudioSegment
import urllib
from PIL import Image, ImageFilter
import random
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import re
import os
import openai
from tenacity import retry, wait_exponential, wait_combine, stop_after_attempt, after_log, before_sleep_log
from diffusers import AudioLDMPipeline
from example_classifications import example_classifications


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import tomesd


# from cldm.model import create_model, load_state_dict
# from cldm.ddim_hacked import DDIMSampler
from laion_face_common import generate_annotation

import subprocess
import json
import glob

from modules.sadtalker_test import SadTalker


# from multiprocessing import Pool


class CustomRootLogger(logging.Logger):
    def setLevel(self, level):
        stack_trace = ''.join(traceback.format_stack())
        print(f"Log level changed to {level} by:\n{stack_trace}")
        super().setLevel(level)


# Replace the root logger with the custom one
logging.setLoggerClass(CustomRootLogger)
root_logger = logging.getLogger()

file_handler = logging.FileHandler(filename='tmp.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logger level

log_format = '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

# Set the formatter and add handlers to the logger
for handler in handlers:
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("logging should be working?")


def custom_exponential_wait(retry_state):
    base_wait = 4
    exponent = 1.2
    return base_wait * (exponent ** retry_state.attempt_number)


def custom_wait_gen():
    attempt = 0
    while True:
        yield custom_exponential_wait(attempt)
        attempt += 1


# from IPython.display import Audio, display


def getFilename(path, extension):
    current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{path}{current_datetime}-{uuid.uuid4()}.{extension}"
    return filename


class AnimeBuilder:

    def __init__(
        self,
        textModel='EleutherAI/gpt-neo-2.7B',
        diffusionModel="hakurei/waifu-diffusion",
        vaeModel="stabilityai/sd-vae-ft-mse",
        templates=templates,
        advanceSceneObjects=None,
        num_inference_steps=30,
        cfg=None,
        verbose=False,
        doImg2Img=False,
        img2imgStrength=0.4,
        saveMemory=True,
        cache_dir='../hf',
        textRevision=None,
        negativePrompt="collage, grayscale, text, watermark, lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, blurry, grayscale, deformed weapons, deformed face, deformed human body",
        suffix=", anime drawing",
        riffusionSuffix=" pleasing rythmic background music",
        savePath="./static/samples/",
        saveImages=False,
        audioLDM="cvssp/audioldm-s-full-v2",
        soundEffectDuration=1.5,
        musicDuration=16,
        musicSuffix=" movie soundtrack background music, smooth jazz",
        imageSizes=[512, 512, 1024, 1024],
        usePITS=True,
        fixAsides=False,
        portraitPrompt=', anime, face, portrait, headshot, white background',
        computeDepth=True,
        osth=True,
        tokenizer=None,
        use_gpt_for_chat_completion=False,
        parallel_screenplays=True,
        controlnet_diffusion_model="runwayml/stable-diffusion-v1-5",
        video_mode=False,
        blur_radius=0.5,
        talking_head_decimate=1,
        face_steps=20,
        max_previous_scenes=6,
        use_GPT4=False,
    ):
        
        self.use_GPT4 = use_GPT4

        self.blur_radius = blur_radius

        self.max_previous_scenes = max_previous_scenes

        self.talking_head_decimate = talking_head_decimate

        self.face_steps = face_steps

        self.saveMemory = saveMemory
        self.doImg2Img = doImg2Img

        # read system prompt files
        self.scenePrompt = open("chapters_to_scenes_systemPrompt.txt").read()
        self.chapterPrompt = open(
            "summary_to_chapters_systemPrompt.txt").read()
        self.screenplayPrompt = open("screenplay_systemPrompt.txt").read()

        self.bonusSceneInstruction = '> NEVER UNDER ANY CIRCUMSTANCES USE THE WORD "MUST"\n\n'

        # load generation functions (for now this is just img2img, move more there later)
        if self.doImg2Img:
            self.generationFunctions = GenerationFunctions(
                saveMemory=self.saveMemory)

        self.video_mode = video_mode

        self.osth = osth
        self.portraitPrompt = portraitPrompt

        self.parallel_screenplays = parallel_screenplays

        # always use parallen when using chatgpt
        if use_gpt_for_chat_completion:
            self.parallel_screenplays = True

        self.fixAsides = fixAsides

        self.imageSizes = imageSizes
        self.img2imgStrength = img2imgStrength

        self.soundEffectDuration = soundEffectDuration
        self.musicDuration = musicDuration
        self.musicSuffix = musicSuffix
        self.savePath = savePath
        self.saveImages = saveImages
        self.use_gpt_for_chat_completion = use_gpt_for_chat_completion

        self.ignored_words = set(
            ["the", "name", "setting", "music", "action", "sound", "effect"])

        self.textModel = textModel

        self.cache_dir = cache_dir

        self.verbose = verbose

        self.mubert = False

        self.templates = templates

        if cfg is None:
            cfg = {
                "genTextAmount_min": 30,
                "genTextAmount_max": 100,
                "no_repeat_ngram_size": 16,
                "repetition_penalty": 1.0,
                "MIN_ABC": 4,
                "num_beams": 1,
                "temperature": 1.0,
                "MAX_DEPTH": 5
            }
        self.cfg = cfg

        self.num_inference_steps = num_inference_steps

        self.negativePrompt = negativePrompt
        self.suffix = suffix
        self.riffusionSuffix = riffusionSuffix

        # use this for advanceScene()
        # advance scene
        if advanceSceneObjects is None:
            self.advanceSceneObjects = [
                {
                    "object": "advancePlot",
                    "whichScene": 3,
                    "numScenes": 3,
                },
                {
                    "object": "fightScene",
                    "whichScene": 1,
                    "numScenes": 3,
                },
            ]
        else:
            self.advanceSceneObjects = advanceSceneObjects

        if self.verbose:
            print("LOADING TEXT MODEL")

        if audioLDM is not None:
            self.audioLDMPipe = AudioLDMPipeline.from_pretrained(
                audioLDM, torch_dtype=torch.float16)
            self.audioLDMPipe = self.audioLDMPipe.to("cuda")

        # move to cpu if saving memory
        if self.saveMemory:
            self.audioLDMPipe = self.audioLDMPipe.to("cpu")

        if self.textModel == "GPT3":
            pass
            # self.textGenerator="GPT3"

            self.textGenerator = {
                'name': "GPT3",
            }

            openai.organization = "org-bKm1yrKncCnPfkcf8pDpe4GM"
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.Model.list()
        elif self.textModel == "gpt-3.5-turbo-instruct":
            # self.textGenerator="gpt-3.5-turbo-instruct"
            self.textGenerator = {
                'name': "gpt-3.5-turbo-instruct",
            }

            openai.organization = "org-bKm1yrKncCnPfkcf8pDpe4GM"
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.Model.list()
        elif self.textModel == "llama":
            thisTokenizer, thisPipeline = getllama()
            self.textGenerator = {
                'name': "llama",
                'tokenizer': thisTokenizer,
                'pipeline': thisPipeline
            }
        else:
            # text model
            self.textModel = textModel
            self.textRevision = textRevision

            textGenerator = pipeline('text-generation',
                                     torch_dtype=torch.float16,
                                     model=self.textModel,
                                     trust_remote_code=True,
                                     device_map="auto",
                                     model_kwargs={"load_in_4bit": True}
                                     )

            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.textModel, torch_dtype=torch.float16)

            else:
                self.tokenizer = tokenizer
                textGenerator.tokenizer = tokenizer

            self.textGenerator = {
                'name': self.textModel,
                'tokenizer': self.tokenizer,
                # 'model': self.textModel
                'pipeline': textGenerator
            }

        # image model

        if self.verbose:
            print("LOADING IMAGE MODEL")

        # make sure you're logged in with `huggingface-cli login`
        # vae = AutoencoderKL.from_pretrained(vaeModel) #maybe I should enable this again?

        # pipe = StableDiffusionPipeline.from_pretrained(diffusionModel,vae=vae, torch_dtype=torch.float16,custom_pipeline="composable_stable_diffusion")
        # pipe = DiffusionPipeline.from_pretrained(
        #    diffusionModel,
        #    vae=vae,
        #    torch_dtype=torch.float16,
        #    custom_pipeline="lpw_stable_diffusion",
        # )

        self.diffusionModel = diffusionModel

        if "xl" in diffusionModel.lower():
            pipe = StableDiffusionXLPipeline.from_single_file(
                diffusionModel, torch_dtype=torch.float16, use_safetensors=True,
                custom_pipeline="lpw_stable_diffusion_xl"
            )
            pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", torch_dtype=torch.float16)

        elif diffusionModel == "LCM":
            pipe = DiffusionPipeline.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main")

            # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
            pipe.to(torch_device="cuda", torch_dtype=torch.float32)

        else:

            # pipe = DiffusionPipeline.from_pretrained(diffusionModel)
            # check if model_id is a .ckpt or .safetensors file
            if diffusionModel.endswith(".ckpt") or diffusionModel.endswith(".safetensors"):
                print("about to die", diffusionModel)
                pipe = StableDiffusionPipeline.from_single_file(diffusionModel,
                                                                torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    diffusionModel, torch_dtype=torch.float16)

            # change to UniPC scheduler
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()
            tomesd.apply_patch(pipe, ratio=0.5)

        self.pipe = pipe

        # if save memory, move pipe to cpu and do garbage collection
        if self.saveMemory:
            self.pipe = self.pipe.to("cpu")
            gc.collect()
            # collect cuda memory
            torch.cuda.empty_cache()
        else:
            self.pipe = self.pipe.to("cuda")

        self.pipe.safety_checker = None

        '''
        if self.doImg2Img:
            if self.verbose:
                print("LOADING Img2Img")

            if "xl" in diffusionModel.lower():
                img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    diffusionModel, torch_dtype=torch.float16, use_safetensors=True)
                # img2img.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
                img2img.enable_vae_tiling()

                self.img2img = img2img

                self.img2img.safety_checker = None

            else:

                if diffusionModel.endswith(".ckpt") or diffusionModel.endswith(".safetensors"):
                    thisModelName = "runwayml/stable-diffusion-v1-5"
                else:
                    thisModelName = diffusionModel

                self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    thisModelName,
                    # revision=revision,
                    scheduler=self.pipe.scheduler,
                    unet=self.pipe.unet,
                    vae=self.pipe.vae,
                    safety_checker=self.pipe.safety_checker,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    torch_dtype=torch.float16,
                    use_auth_token=True,
                    cache_dir="./AI/StableDiffusion"
                )

                self.img2img.enable_attention_slicing()
                self.img2img.enable_xformers_memory_efficient_attention()
                tomesd.apply_patch(self.img2img, ratio=0.5)

            # if save memmory, move to cpu and do garbage collection
            if self.saveMemory:
                self.img2img = self.img2img.to("cpu")
                gc.collect()
                # collect cuda memory
                torch.cuda.empty_cache()
            '''

        if self.verbose:
            print("LOADING TTS MODEL")

        # tts
        #

        self.usePITS = usePITS
        if usePITS:
            self.pitsTTS = pits.GradioApp(pits.get_default_args())

        else:

            models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
                "facebook/fastspeech2-en-200_speaker-cv4",  # random
                arg_overrides={"vocoder": "hifigan", "fp16": False, }
            )

            self.tts_models = models
            self.tts_cfg = cfg
            self.tts_task = task

            # model = models[0]
            TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
            self.tts_generator = task.build_generator(models, cfg)

            #  000000000011111111112222222222333333333344444444444555555555
            #  012345678901234567890123456789012345678901234567890123456789

        if self.usePITS:
            #    01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234501234567890
            #    00000000001111111111222222222233333333334444444444555555555566666666667777777777888888888899999999990000000000
            s = "fmmffffmfffmfffmmfmmmffffmfmfmfmmmffmffffffmmmmmmffmmmffmmmmfmffffmfffmfmfffffmfffmfffmfffmffffffmmfmffmmmmf".upper()
        else:
            s = "FMFMMMMMFMMMFFMFFMMMMMMmffmmfmmfmfmmmmmmfmmmmmfmmmffmmmm".upper()

        self.maleVoices = [i for i in range(len(s)) if s[i] == "M"]
        self.femaleVoices = [i for i in range(len(s)) if s[i] == "F"]

        # controlnet for portrait generation
        # self.facemodel = create_model('../cldm_v21.yaml').cpu()
        # self.facemodel.load_state_dict(load_state_dict(
        #    '..\ControlNet\models/controlnet_sd21_laion_face_v2_full.ckpt', location='cuda'))
        # self.facemodel = self.facemodel.cuda()
        # self.facemodel = self.facemodel.cpu()
        # self.facemodel_ddim_sampler = DDIMSampler(self.facemodel)  # ControlNet _only_ works with DDIM.

        # Stable Diffusion 2.1-base:
        # controlnet = ControlNetModel.from_pretrained(
        #    "CrucibleAI/ControlNetMediaPipeFace", torch_dtype=torch.float16, variant="fp16")
        # self.facepipe = StableDiffusionControlNetPipeline.from_pretrained(
        #    "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        # )

        # controlnet = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", subfolder="diffusion_sd15")
        # if diffusionModel.endswith(".ckpt") or diffusionModel.endswith(".safetensors"):
        #    self.facepipe = StableDiffusionControlNetPipeline.from_single_file(diffusionModel, controlnet=controlnet, safety_checker=None)
        # else:
        #    self.facepipe = StableDiffusionControlNetPipeline.from_pretrained(diffusionModel, controlnet=controlnet, safety_checker=None)

        # self.facepipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # Remove if you do not have xformers installed
        # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
        # for installation instructions
        # self.facepipe.enable_xformers_memory_efficient_attention()
        # self.facepipe.enable_model_cpu_offload()

        if "xl" in diffusionModel.lower():
            # TODO: add sdxl controlnet when it's available
            pass

        # OR
        # Stable Diffusion 1.5:
        controlnet = ControlNetModel.from_pretrained(
            "CrucibleAI/ControlNetMediaPipeFace", subfolder="diffusion_sd15", torch_dtype=torch.float16, variant="fp16")

        if "safetensors" in controlnet_diffusion_model:
            self.facepipe = StableDiffusionControlNetPipeline.from_single_file(
                controlnet_diffusion_model, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
        else:

            self.facepipe = StableDiffusionControlNetPipeline.from_pretrained(
                controlnet_diffusion_model, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)

        # disable safety checker
        self.facepipe.safety_checker = None

        self.facepipe.scheduler = UniPCMultistepScheduler.from_config(
            self.facepipe.scheduler.config)

        # Remove if you do not have xformers installed
        # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
        # for installation instructions
        self.facepipe.enable_xformers_memory_efficient_attention()
        # self.facepipe.enable_model_cpu_offload()

        # if save memmory, move to cpu and do garbage collection
        if self.saveMemory:
            self.facepipe = self.facepipe.to("cpu")
            gc.collect()
            # collect cuda memory
            torch.cuda.empty_cache()
        else:
            self.facepipe = self.facepipe.to("cuda")

        if not self.osth:
            self.sad_talker = SadTalker("E:\img\SadTalker")

        if computeDepth:
            repo = "isl-org/ZoeDepth"
            # Zoe_N
            model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
            DEVICE = "cuda"
            self.zoe = model_zoe_n.to(DEVICE)
        else:
            self.zoe = None

        if self.saveMemory:
            self.zoe = self.zoe.to("cpu")
            gc.collect()
            # collect cuda memory
            torch.cuda.empty_cache()
        else:
            self.zoe = self.zoe.to("cuda")

    def chatCompletion(self, messages, n=1, min_new_tokens=256, max_new_tokens=512, generation_prefix=""):

        # free up some memory
        gc.collect()
        torch.cuda.empty_cache()

        # first we need to combine messages into a single string
        # as a reminder messages have the format {"role": "system/user/assistant", "content": "this is some conent"}
        prompt = ""
        lastRole = "system"
        for message in messages:
            # prompt += message['role']+":\n"
            if message['role'] != lastRole:
                prompt += "\n"
            prompt += message['content']+"\n"
            lastRole = message['role']

        # now add a final "assitant:" to the prompt
        # prompt += "assistant:\n"
        # now we can run the completion

        prompt += "\n"+generation_prefix

        output = []
        for i in range(n):

            # print("\n=====\n", prompt, "\n=====\n")

            result = self.textGenerator['pipeline'](prompt,
                                                    min_new_tokens=min_new_tokens,
                                                    max_new_tokens=max_new_tokens,
                                                    return_full_text=True,
                                                    no_repeat_ngram_size=self.cfg["no_repeat_ngram_size"],
                                                    repetition_penalty=self.cfg["repetition_penalty"],
                                                    num_beams=self.cfg["num_beams"],
                                                    temperature=self.cfg["temperature"],
                                                    do_sample=True,
                                                    )

            result_text = result[0]['generated_text']

            # print("\n=====\n", result_text, "\n=====\n")

            # now we need to pull out the resulting message
            start_index = len(prompt)

            # stop at \n\n
            end_index = result_text.find("\n\n", start_index)

            # end_index = result_text.find("user:", start_index)

            # print("start_index:", start_index, "end_index:", end_index)

            output += [generation_prefix+result_text[start_index:end_index]]
            # output += [generation_prefix+result_text]

        # free up some memory
        gc.collect()
        torch.cuda.empty_cache()

        return output

    def _get_portrait0(self, input_image: Image.Image, prompt, a_prompt, n_prompt, max_faces, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta):

        # move to cuda
        self.facemodel = self.facemodel.cuda()

        # ControlNet _only_ works with DDIM.
        facemodel_ddim_sampler = DDIMSampler(self.facemodel)

        with torch.no_grad():
            empty = generate_annotation(input_image, max_faces)
            visualization = Image.fromarray(empty)  # Save to help debug.

            empty = numpy.moveaxis(empty, 2, 0)  # h, w, c -> c, h, w
            control = torch.from_numpy(empty.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            # control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            # Sanity check the dimensions.
            B, C, H, W = control.shape
            assert C == 3
            assert B == num_samples

            if seed != -1:
                random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                numpy.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True

            # if config.save_memory:
            #    model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [
                self.facemodel.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [
                self.facemodel.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            # if config.save_memory:
            #    model.low_vram_shift(is_diffusing=True)

            self.facemodel.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = facemodel_ddim_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond
            )

            # if config.save_memory:
            #    model.low_vram_shift(is_diffusing=False)

            x_samples = self.facemodel.decode_first_stage(samples)
            # x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(numpy.uint8)
            x_samples = numpy.moveaxis((x_samples * 127.5 + 127.5).cpu().numpy().clip(
                0, 255).astype(numpy.uint8), 1, -1)  # b, c, h, w -> b, h, w, c
            results = [visualization] + [x_samples[i]
                                         for i in range(num_samples)]

        # move to cpu
        # move to cuda
        self.facemodel = self.facemodel.to('cuda')
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def _get_portrait(self, input_image: Image.Image, prompt, a_prompt, n_prompt, NUM_RETRIES=3):
        empty = generate_annotation(input_image, 1)
        anno = Image.fromarray(empty).resize((768, 768))

        # if save memory, move from cpu to gpu
        if self.saveMemory:
            self.facepipe = self.facepipe.to('cuda')

        image = self.facepipe(prompt+a_prompt, negative_prompt=n_prompt,
                              image=anno, num_inference_steps=self.face_steps).images[0]
        # image = self.facepipe(prompt+a_prompt, negative_prompt=n_prompt,
        #                      image=input_image, num_inference_steps=30).images[0]

        # check if image is all black, and if so, retry
        for i in range(NUM_RETRIES):
            if np.all(np.array(image) == 0):
                print("RETRYING PORTRAIT")
                image = self.facepipe(prompt+a_prompt, negative_prompt=n_prompt,
                                      image=anno, num_inference_steps=self.face_steps).images[0]
            else:
                break

        # if save memory, move from gpu to cpu
        if self.saveMemory:
            self.facepipe = self.facepipe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

        image.save("./static/samples/tmp.png")

        return image

    def getPortrait(self, prompt, promptSuffix, img2imgStrength=0.6, num_inference_steps=20):

        depth_image_path = "./nan3.jpg"

        input_image = Image.open(depth_image_path)

        input_image = input_image.resize((512, 512))

        # a_prompt=',anime, face, portrait, headshot, white background'#added to prompt
        a_prompt = self.portraitPrompt+promptSuffix
        # n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'#negative prompt
        n_prompt = "hands, watermark, "+self.negativePrompt
        max_faces = 1
        num_samples = 1
        ddim_steps = 10
        guess_mode = False
        strength = 0.8
        scale = 7.5  # cfg scale
        seed = np.random.randint(0, 10000)
        eta = 0

        print("creating portrait with prompt:", prompt+a_prompt)

        # results = self._get_portrait(input_image, prompt, a_prompt, n_prompt, max_faces,
        #                             num_samples, ddim_steps, guess_mode, strength, scale, seed, eta)
        # results = self._get_portrait(input_image, prompt, a_prompt, n_prompt)
        # output = Image.fromarray(results[1])
        output = self._get_portrait(input_image, prompt, a_prompt, n_prompt)

        if self.doImg2Img:

            # img2Input = output.resize((self.imageSizes[2], self.imageSizes[3]))
            img2Input = output.resize((1024, 1024))

            '''

            # some nonsense to handle long prompts, based off of https://github.com/huggingface/diffusers/issues/2136#issuecomment-1409978949
            # todo figure out what this
            max_length = self.pipe.tokenizer.model_max_length

            input_ids = self.pipe.tokenizer(
                prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to("cuda")

            # negative_ids = self.pipe.tokenizer(self.negativePrompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids
            negative_ids = self.pipe.tokenizer(
                self.negativePrompt, truncation=True, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids
            negative_ids = negative_ids.to("cuda")

            padding_length = max_length - (input_ids.shape[-1] % max_length)

            if padding_length > 0:
                input_ids = torch.cat([input_ids, torch.full((input_ids.shape[0], padding_length),
                                      self.pipe.tokenizer.pad_token_id, dtype=torch.long, device="cuda")], dim=1)
                negative_ids = torch.cat([negative_ids, torch.full(
                    (negative_ids.shape[0], padding_length), self.pipe.tokenizer.pad_token_id, dtype=torch.long, device="cuda")], dim=1)

            concat_embeds = []
            neg_embeds = []
            for i in range(0, input_ids.shape[-1], max_length):
                concat_embeds.append(self.pipe.text_encoder(
                    input_ids[:, i: i + max_length])[0])
                neg_embeds.append(self.pipe.text_encoder(
                    negative_ids[:, i: i + max_length])[0])

            prompt_embeds = torch.cat(concat_embeds, dim=1)
            negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

            

            if self.saveMemory:
                self.img2img = self.img2img.to('cuda')

            # with autocast("cuda"):
            if True:  # for some reason autocast is bad?
                img2 = self.img2img(
                    prompt=prompt,
                    negative_prompt=self.negativePrompt,
                    # prompt_embeds=prompt_embeds,
                    # negative_prompt_embeds=negative_prompt_embeds,
                    image=img2Input,
                    strength=img2imgStrength,
                    guidance_scale=7.5,
                    num_inference_steps=num_inference_steps,
                ).images[0]
                output = img2

            if self.saveMemory:
                self.img2img = self.img2img.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()

            '''
            img2 = self.generationFunctions.image_to_image(img2Input,
                                                           prompt,
                                                           "low resolution, blurry, "+self.negativePrompt,
                                                           img2imgStrength,
                                                           steps=num_inference_steps)

            output = img2

        # return output
        filename = getFilename(self.savePath, "png")
        output.save(filename)

        if self.zoe is not None:
            depthFilename = filename.replace(".png", "_depth.png")
            depth = self.getZoeDepth(output)
            depth.save(depthFilename)

        print("DIED")

        return filename

    def getTalkingHeadVideo(self, portrait_image_path, text, voice, gender, supress=True, decimate=1):

        audio_file_path, duration = self.textToSpeech(text, voice, gender)

        # make sure audio_file_path ends with .wav (this file exists either way)
        if not audio_file_path.endswith('.wav'):
            audio_file_path = audio_file_path[:-4]+".wav"

        if self.osth:

            image_path = portrait_image_path
            save_dir = getFilename(self.savePath, "mov")

            if image_path.endswith('.png'):
                png_path = os.path.join(image_path)
                jpg_path = os.path.join(
                    os.path.splitext(image_path)[0] + '.jpg')
                img = Image.open(png_path)
                rgb_img = img.convert('RGB')
                rgb_img.save(jpg_path)
                image_path = jpg_path

            osth_path = '.\AAAI22-one-shot-talking-face'

            os.makedirs(save_dir, exist_ok=True)
            phoneme = processAudio(
                audio_file_path, phindex_location=".\AAAI22-one-shot-talking-face\phindex.json")

            # supress printing
            if supress == True:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    mov = test_with_input_audio_and_image(image_path, audio_file_path, phoneme,
                                                          ".\\AAAI22-one-shot-talking-face\\checkpoint\\generator.ckpt",
                                                          ".\\AAAI22-one-shot-talking-face\\checkpoint\\audio2pose.ckpt",
                                                          save_dir, osth_path, decimate=decimate)
            else:
                mov = test_with_input_audio_and_image(image_path, audio_file_path, phoneme,
                                                      ".\\AAAI22-one-shot-talking-face\\checkpoint\\generator.ckpt",
                                                      ".\\AAAI22-one-shot-talking-face\\checkpoint\\audio2pose.ckpt",
                                                      save_dir, osth_path, decimate=decimate)

            print(mov)
            found_movie = glob.glob(os.path.join(save_dir, "*.mp4"))

            return found_movie[0], duration

        else:
            # use sadtalker
            driven_audio = audio_file_path
            source_image = portrait_image_path
            still_mode = False
            resize_mode = True
            use_enhancer = False
            result_dir = ".\static\samples"
            result = self.sad_talker.test(
                source_image,
                driven_audio,
                still_mode,
                resize_mode,
                use_enhancer,
                result_dir
            )

            # replace all #'s with _'s in filename
            newFilename = result[0].replace("#", "_")
            os.rename(result[0], newFilename)

            return newFilename, duration

    def doGen(self, prompt, num_inference_steps=30, recursion=0):

        # move text model to cpu for now
        # if self.saveMemory:
        #    self.textGenerator['pipeline'].model = self.textGenerator['pipeline'].model.cpu(
        #    )
        #    gc.collect()
        #    torch.cuda.empty_cache()

        seed = np.random.randint(0, 1000000)
        print("SEED: ", seed, "")

        generator = torch.Generator("cuda").manual_seed(seed)

        print("ABOUT TO DIE")

        # if save memory, move out of cpu
        if self.saveMemory:
            self.pipe = self.pipe.to('cuda')

        if self.diffusionModel == "LCM":

            image = self.pipe([prompt],
                              # negative_prompt=[self.negativePrompt], #not supported for some reason :(
                              guidance_scale=7.5,
                              num_inference_steps=num_inference_steps,
                              width=self.imageSizes[0],
                              height=self.imageSizes[1],
                              # generator=generator
                              ).images[0]

        else:
            with autocast("cuda"):
                image = self.pipe([prompt],
                                  negative_prompt=[self.negativePrompt],
                                  guidance_scale=7.5,
                                  num_inference_steps=num_inference_steps,
                                  width=self.imageSizes[0],
                                  height=self.imageSizes[1],
                                  generator=generator
                                  ).images[0]

        # if save memory, move back to cpu
        if self.saveMemory:
            self.pipe = self.pipe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

        print("DIED")
        image.save("./static/samples/test.png")

        if self.doImg2Img:

            # low pass filter
            blurred_image = image.filter(
                ImageFilter.GaussianBlur(radius=self.blur_radius))
            img2Input = blurred_image.resize(
                (self.imageSizes[2], self.imageSizes[3]))

            # img2Input = image.resize((self.imageSizes[2], self.imageSizes[3]))

            '''

            # move img2img model to gpu for now
            if self.saveMemory:
                self.img2img = self.img2img.to('cuda')

            # with autocast("cuda"):
            if True:
                img2 = self.img2img(
                    prompt=prompt,
                    negative_prompt=self.negativePrompt,
                    # prompt_embeds=prompt_embeds,
                    # negative_prompt_embeds=negative_prompt_embeds,
                    image=img2Input,
                    strength=self.img2imgStrength,
                    guidance_scale=7.5,
                    num_inference_steps=num_inference_steps,
                ).images[0]
                output = img2

                img2.save("./static/samples/test2.png")

            # move img2img model back to cpu
            if self.saveMemory:
                self.img2img = self.img2img.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()

            print("DIED2")

            '''
            img2 = self.generationFunctions.image_to_image(img2Input,
                                                           prompt,
                                                           self.negativePrompt,
                                                           self.img2imgStrength,
                                                           steps=num_inference_steps)

            output = img2

        else:
            output = image

        if self.saveMemory:
            gc.collect()
            torch.cuda.empty_cache()

            # self.textGenerator['pipeline'].model = self.textGenerator['pipeline'].model.cuda()

        # fix all black images? (which Anything 3.0 puts out sometimes)
        pix = np.array(output)
        MAX_IMG_RECURSION = 3
        if np.sum(pix) == 0 and recursion < MAX_IMG_RECURSION:
            if self.verbose:
                print("REDOING BLANK IMAGE!")
            return self.doGen(prompt, num_inference_steps, recursion=recursion+1)

        # return output
        # convert to file and return
        filename = getFilename(self.savePath, "png")

        if self.zoe is not None:
            depthFilename = filename.replace(".png", "_depth.png")
            depth = self.getZoeDepth(output)
            depth.save(depthFilename)

        output.save(filename)
        return filename

    def getZoeDepth(self, image, boxSize=1, blurRadius=1):

        if self.saveMemory:
            self.zoe = self.zoe.to('cuda')

        depth = self.zoe.infer_pil(image)  # as numpy

        if self.saveMemory:
            self.zoe = self.zoe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

        value = depth
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        v2 = 1-value/np.min(value)
        value = (boxSize+v2)/boxSize
        value = value.squeeze()
        # crop to 0-1
        value = np.clip(value, 0, 1)
        formatted = (value * 255 / np.max(value)).astype('uint8')
        img = Image.fromarray(formatted)
        img = img.filter(
            ImageFilter.GaussianBlur(radius=blurRadius))
        return img

    def generateAudio(self, prompt, duration=3, num_inference_steps=10):
        mp3file_name = getFilename(self.savePath, "mp3")
        # wavfile_name = getFilename(self.savePath, "wav")
        wavfile_name = mp3file_name.replace(".mp3", ".wav")

        # if save memory, move out of cpu
        if self.saveMemory:
            self.audioLDMPipe = self.audioLDMPipe.to('cuda')

        audio = self.audioLDMPipe(
            prompt, num_inference_steps=num_inference_steps, audio_length_in_s=duration).audios[0]

        # if save memory, move back to cpu
        if self.saveMemory:
            self.audioLDMPipe = self.audioLDMPipe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

        audio = ipd.Audio(audio, rate=16000, autoplay=True)
        with open(wavfile_name, 'wb') as f:
            f.write(audio.data)

        wavfile = AudioSegment.from_wav(wavfile_name)
        wavfile.export(mp3file_name, format="mp3")

        return mp3file_name, duration

    def textToSpeech(self, text, voice, gender):

        print("doing tts, voice=", voice, gender)

        mp3file_name = getFilename(self.savePath, "mp3")
        # wavfile_name = getFilename(self.savePath, "wav")
        wavfile_name = mp3file_name.replace(".mp3", ".wav")

        if self.usePITS:

            scope_shift = 0

            # if gender=="male":
            #    scope_shift=10
            # elif gender=="female":
            #    scope_shift=-10

            duration_shift = 1.0
            seed = 1

            ph, (rate, wav) = self.pitsTTS.inference(
                text, voice, seed, scope_shift, duration_shift)

            # pad wav with "rate" zeros to make it 1 second longer
            wav = np.pad(wav, (0, rate), mode="constant")

            audio = ipd.Audio(wav, rate=rate, autoplay=True)
            with open(wavfile_name, 'wb') as f:
                f.write(audio.data)

            duration = len(wav)/rate

            wavfile = AudioSegment.from_wav(wavfile_name)
            wavfile.export(mp3file_name, format="mp3")

            print("done tts")

            return mp3file_name, duration

        else:

            try:
                with autocast("cuda"):
                    self.tts_task.data_cfg.hub["speaker"] = voice
                    sample = TTSHubInterface.get_model_input(
                        self.tts_task, text)
                    # print("about to die",models[0],sample)
                    wav, rate = TTSHubInterface.get_prediction(
                        self.tts_task, self.tts_models[0], self.tts_generator, sample)
                    # print("huh?",wav,rate,len(wav)/rate)
                    duration = len(wav)/rate

                audio = ipd.Audio(wav.cpu(), rate=rate, autoplay=True)
                with open(wavfile_name, 'wb') as f:
                    f.write(audio.data)

                wavfile = AudioSegment.from_wav(wavfile_name)
                wavfile.export(mp3file_name, format="mp3")

                return mp3file_name, duration
            except:
                print("Error generating text", text, voice)
        # music

    def generate_track_by_prompt_vol(self, prompt, vol=1.0, duration=8, loop=True, autoplay=True):

        # if self.audioLDMPipe is not None:
        filename, duration = self.generateAudio(prompt, duration=duration)
        return filename

        mp3file_name = getFilename(self.savePath, "mp3")
        wavfile_name = getFilename(self.savePath, "wav")

        if self.mubert:

            url = generate_track_by_prompt(prompt, duration, loop)
            if url is None:
                return
            mp3 = urllib.request.urlopen(url).read()
            original = AudioSegment.from_mp3(BytesIO(mp3))
            samples = original.get_array_of_samples()
            samples /= np.max(np.abs(samples))
            samples *= vol
            # audio = Audio(samples, normalize=False,
            #              rate=original.frame_rate, autoplay=autoplay)

            # audio = Audio.from_file("audio.mp3", loop=True, autoplay=True)

            # return audio
            return mp3file_name
        else:
            _, filename = get_music(prompt+self.riffusionSuffix, duration,
                                    wavfile_name=wavfile_name, mp3file_name=mp3file_name)
            mp3 = open(filename, 'rb').read()
            original = AudioSegment.from_mp3(BytesIO(mp3))
            samples = original.get_array_of_samples()
            samples /= np.max(np.abs(samples))
            samples *= vol
            # audio = Audio(samples, normalize=False,
            #              rate=original.frame_rate, autoplay=autoplay)
            # audio = Audio.from_file("audio.mp3", loop=True, autoplay=True)

            # return audio
            return mp3file_name

    def descriptionToCharacter(self, description):
        thisObject = WorldObject(self.templates, self.textGenerator, "descriptionToCharacter", objects={
            "description": description},
            cfg=self.cfg,
            verbose=self.verbose
        )
        return thisObject

    def advanceStory(self, story, subplot, mainCharacter=None, supportingCharacters=None, alwaysUseMainCharacter=True):

        # save some memory
        self.pipe.to("cpu")
        riffusion.pipe2.to('cpu')

        gc.collect()
        torch.cuda.empty_cache()

        advanceSceneObject = random.choice(self.advanceSceneObjects)

        # update subplot

        if alwaysUseMainCharacter:
            character1 = mainCharacter
            character2 = random.choice(supportingCharacters)
        else:
            character1, character2 = random.sample(
                [mainCharacter]+supportingCharacters, 2)

        if character1 is None:
            character1 = story.getProperty("character1")
        if character2 is None:
            character2 = story.getProperty("character2")

        newStory = WorldObject(self.templates, self.textGenerator, advanceSceneObject['object'], objects={
            "character1": character1,
            "character2": character2,
            "previous": story,
        },
            cfg=self.cfg,
            verbose=self.verbose
        )

        whichScene = advanceSceneObject['whichScene']
        numScenes = advanceSceneObject['numScenes']

        self.pipe.to("cuda")
        riffusion.pipe2.to('cuda')

        gc.collect()
        torch.cuda.empty_cache()

        return whichScene, numScenes, newStory

    def sceneToTranscript(self, scene, k=3, character1=None, character2=None, whichScene=1):
        if character1 is None:
            character1 = scene.getProperty("character1")
        if character2 is None:
            character2 = scene.getProperty("character2")

        objects = {"story synopsis": scene.getProperty("story synopsis"),
                   "subplot": scene.getProperty("subplot"),
                   "scene": scene.getProperty("scene %d" % k),
                   "character1": character1,
                   "character2": character2,
                   }

        # check for dialogue
        line1txt = None
        try:
            line1txt = scene.getProperty("scene %d line 1 text" % whichScene)
            if self.verbose:
                print("line 1 text", line1txt)
        except:
            if self.verbose:
                print("no property", "scene %d line 1 text" % whichScene)
            pass
        if line1txt:
            objects['line 1 text'] = line1txt

        thisObject = WorldObject(self.templates, self.textGenerator,
                                 "sceneToTranscript", objects,
                                 cfg=self.cfg,
                                 verbose=self.verbose
                                 )
        return thisObject

    def watchAnime(
        self,
        synopsis=None,
        subplot1=None,
        scene1=None,
        character1=None,
        num_characters=4,
        k=100,
        amtMin=15,
        amtMax=30,
        promptSuffix="",
        portrait_size=128,
        skip_transcript=False,
        whichScene=1,  # optionally skip first few scenes
        alwaysUseMainCharacter=True,  # always use main character in scene
    ):

        # make sure text generator is on cuda (can get out of sync if we ctrl+c during doGen() )
        # if self.textGenerator["name"].startswith("GPT3"):
        #    self.textGenerator['pipeline'].model = self.textGenerator['pipeline'].model.cuda(
        #    )

        self.amtMin = amtMin
        self.amtMax = amtMax

        objects = {}
        if synopsis:
            objects['story synopsis'] = synopsis
        if scene1:
            objects['scene 1 text'] = scene1
        if character1:
            if isinstance(character1, str):
                character1 = self.descriptionToCharacter(character1)
                # print(character1)
        else:
            character1 = WorldObject(self.templates, self.textGenerator, "character",
                                     cfg=self.cfg,
                                     verbose=self.verbose
                                     )

        mainCharacter = character1
        objects['character1'] = mainCharacter
        if self.verbose:
            print("main character", mainCharacter.__repr__())

        names = set()
        names.add(str(mainCharacter.getProperty("name")))
        # generate characters
        supportingCharacters = []
        while len(supportingCharacters) < num_characters-1:
            newCharacter = WorldObject(self.templates, self.textGenerator, "character",
                                       cfg=self.cfg,
                                       verbose=self.verbose
                                       )
            thisName = str(newCharacter.getProperty("name"))
            if thisName not in names:
                if self.verbose:
                    print(newCharacter.__repr__())
                supportingCharacters += [newCharacter]
                names.add(thisName)
            else:
                if self.verbose:
                    print("skipping repeated character", thisName)

        if subplot1:
            objects['part 1'] = subplot1

        for i in range(3):
            objects['character%d' % (i+2)] = supportingCharacters[i]

        plotOverview = WorldObject(
            self.templates, self.textGenerator, "plot overview",
            cfg=self.cfg,
            verbose=self.verbose
        )

        subplot = plotOverview.getProperty("part 1")
        objects["subplot"] = subplot

        story = WorldObject(self.templates, self.textGenerator,
                            "storyWithCharacters",
                            cfg=self.cfg,
                            objects=objects,
                            verbose=self.verbose
                            )
        if self.verbose:
            print(story)

        # get voices
        voices = {}
        genders = {}
        for thisCharacter in [mainCharacter]+supportingCharacters:
            name = str(thisCharacter.getProperty("name"))
            gender = thisCharacter.getProperty("gender")

            if self.usePITS:
                # voices[name]=random.randint(0,len(self.pitsTTS.hps.data.speakers)-1)
                if gender == "male":
                    voices[name] = random.choice(self.maleVoices)
                else:
                    voices[name] = random.choice(self.femaleVoices)
            else:
                if gender == "male":
                    voices[name] = random.choice(self.maleVoices)
                else:
                    voices[name] = random.choice(self.femaleVoices)
            genders[name] = gender

            print("GOT GENDER FOR:", name, "=", gender)

            description = thisCharacter.getProperty("description")

        # generate portraits
        portraits = {}
        for thisCharacter in [mainCharacter]+supportingCharacters:
            name = str(thisCharacter.getProperty("name"))
            gender = thisCharacter.getProperty("gender")
            description = thisCharacter.getProperty("description")
            prompt = "high resolution color portrait photograph of "+gender+", "+description + \
                ", solid white background"+promptSuffix
            portrait = self.doGen(
                prompt, num_inference_steps=self.num_inference_steps)
            portraits[name] = portrait
            yield {"debug": description}
            yield {"image": portrait,
                   "width": 1024,
                   "height": 1024}

        synopsis = story.getProperty("story synopsis")

        whichScene = whichScene
        numScenes = 3
        for i in range(k):

            scene = str(story.getProperty("scene %d" % whichScene))
            whichSubplot = (i*5//k)+1
            wss = "part %d" % whichSubplot
            thisSubplot = plotOverview.getProperty(wss)
            story.objects['subplot'] = thisSubplot

            audio = self.generate_track_by_prompt_vol(
                scene, vol=0.25, duration=self.musicDuration)

            # parse out character1 and character2
            character1 = None
            for this_character1 in [mainCharacter]+supportingCharacters:
                if str(this_character1.getProperty("name")) in scene:
                    character1 = this_character1
                    character1description = character1.getProperty(
                        "description")
                    break
            character2 = None
            for this_character2 in [mainCharacter]+supportingCharacters:
                # gah, bug was that we were finding the same person twice!
                if character1 is not None and str(this_character2.getProperty("name")) == str(character1.getProperty("name")):
                    continue
                if str(this_character2.getProperty("name")) in scene:
                    character2 = this_character2
                    character2description = character2.getProperty(
                        "description")
                    break

            # swap order if needed
            if character1 is not None and character2 is not None:
                name1 = str(character1.getProperty("name"))
                name2 = str(character2.getProperty("name"))
                i1 = scene.index(name1)
                i2 = scene.index(name2)

                # print("indexes", i1, i2)

                if i1 > i2:
                    # swap
                    character1, character2 = character2, character1
                    character1description, character2description = character2description, character1description
            else:
                # print("huh?", character1, character2)
                pass

            prompt = scene + ", "
            if character1 is not None:
                prompt += character1description
            else:
                print("Error, could not find character1", scene)

            if character2 is not None:
                prompt += ", "+character2description+","+promptSuffix
            else:
                print("Error, could not find character2", scene)

            image = self.doGen(
                prompt, num_inference_steps=self.num_inference_steps)

            yield {"debug": "Subplot: %s\n Scene: %s" % (thisSubplot, scene)}
            if audio:
                yield {"music": audio}
            else:
                print("err, no music!")

            if self.doImg2Img:
                width, height = self.imageSizes[2], self.imageSizes[3]
            else:
                width, height = self.imageSizes[2], self.imageSizes[3]

            yield {"image": image,
                   "width": width,
                   "height": height,
                   }
            transcript = self.sceneToTranscript(
                story, k=whichScene,
                character1=character1,
                character2=character2,
                whichScene=whichScene,
            )

            if self.verbose:
                print(transcript)

            # generate dialogue
            if skip_transcript == False:
                tt = transcript.getProperty("transcript")
                for line in tt.split("\n"):
                    # thisImg = image.copy()
                    name, dialogue = line.split(":")
                    voice = voices[name]
                    portrait = portraits[name]
                    gender = genders[name]

                    try:

                        speech, duration = self.getTalkingHeadVideo(
                            portrait, dialogue, voice, gender, decimate=self.talking_head_decimate)

                    except Exception as e:
                        traceback.print_exc()
                        print("Error generating talking head video:", e)
                        return None

                    yield {"speech": speech,
                           "duration": duration+1,
                           "name": name,
                           "dialogue": dialogue}

            # advance plot if necessary
            whichScene += 1
            if whichScene > numScenes:
                whichScene, numScenes, story = self.advanceStory(
                    story,
                    thisSubplot,
                    mainCharacter=mainCharacter,
                    supportingCharacters=supportingCharacters,
                    alwaysUseMainCharacter=alwaysUseMainCharacter
                )
                if self.verbose:
                    print("advancing scene", story, whichScene, numScenes)
            else:
                # print("not advancing",whichScene,numScenes)
                pass

    def getTagBundles(self, longscreenplay):

        tags = set([x.split(":")[0].lower()
                   for x in longscreenplay.split("\n") if ":" in x])

        tags = [x for x in tags if len(x.split()) < 4]

        tag_bundles = []

        for tag in tags:
            tagset = set(tag.split())-self.ignored_words
            if len(tagset) == 0:
                continue
            t = 0
            for bundle in tag_bundles:
                if tagset.intersection(bundle):
                    t = 1
                    bundle.update(tagset)
            if t == 0:
                tag_bundles += [tagset]

        # print(tag_bundles)

        # and let's do that more more time

        new_tag_bundles = []

        for tagset in tag_bundles:
            t = 0
            for bundle in new_tag_bundles:
                if tagset.intersection(bundle):
                    t = 1
                    bundle.update(tagset)
            if t == 0:
                new_tag_bundles += [tagset]

        # print(new_tag_bundles)

        return new_tag_bundles

    def normalizeTag(self, tag, tag_bundles):
        tagset = set(tag.split())-self.ignored_words
        if len(tagset) == 0:
            print("this should never happen!")
            return tag
        t = 0
        for bundle in tag_bundles:
            if tagset.intersection(bundle):
                return "_".join(bundle)
        print("this should never happen!")
        return tag

    def mergeName0(self, name1, names):
        s1 = set(name1.lower().split())-self.ignored_words
        # @s1 = set([x for x in s1 if len(x) > 3]) #not actually helpful
        for name2 in names:
            s2 = set(name2.lower().split())-self.ignored_words
            if s1.intersection(s2):
                # don't merge if they contain different digits
                digits = set([str(i) for i in range(10)])
                if len(s1.intersection(digits)) > 0 and len(s2.intersection(digits)) > 0:
                    if s1.intersection(digits) != s2.intersection(digits):
                        continue
                return name2
        return name1

    def mergeName(self, name1, names):
        s1 = set(name1.lower().split())-self.ignored_words
        # s1 = set([x for x in s1 if len(x) > 3])#not actually helpful
        if len(s1) == 0:
            return name1
        for name2 in names:
            s2 = set(name2.lower().split())-self.ignored_words
            # s2 = set([x for x in s2 if len(x) > 3])
            if len(s2) == 0:
                continue
            if s1.issubset(s2):
                return name2
            if s1.issuperset(s2):
                return name2
        return name1

    def enhancePrompt(self, prompt, characters, storyObjects=None):
        output = prompt
        didEnhance = False

        print("ABOUT TO DIE")
        try:
            if storyObjects is not None and storyObjects.has("prompt object"):
                prompt_object = storyObjects.getProperty("prompt object")
                promptObject = WorldObject(
                    self.templates,
                    self.textGenerator,
                    prompt_object,
                    verbose=False,
                    cfg=self.cfg
                )
                print("GOT PROMPT OBJECT:", promptObject)
                didEnhance = True
                output += ", " + str(promptObject)
        except Exception as e:
            traceback.print_exc()
            print("ERROR ENHANCING PROMPT:", e)

        for name in characters.keys():
            n = set([w.lower()
                    for w in name.split() if len(w) > 2])-self.ignored_words
            for w in n:
                if w in prompt:
                    output += " "+characters[name].getProperty("description")
                    didEnhance = True
                    break

        return output, didEnhance

    def generateNewCharacter(self, tag, _characters):

        # create a custom template where we add the existing characters to the template

        customTemplate = self.templates["character"]

        # split on \n\n
        customTemplate = customTemplate.split("\n\n")

        # filter out anything thats just whitespace
        customTemplate = [x for x in customTemplate if x.strip() != ""]

        # print("CUSTOM TEMPLATE ENDS WITH\n--\n"+customTemplate[-1])

        # add the filled templates of the existing characters before the final template
        for character in _characters.values():
            character_repr = character.__repr__()
            # remove lines that start with <
            character_repr = "\n".join(
                [x for x in character_repr.split("\n") if not x.startswith("<")])
            customTemplate.insert(-1, character_repr)

        # join the templates back together
        customTemplate = "\n\n".join(customTemplate)

        # debug
        # print("CREATING NEW CHARACTER",tag+"\n===\n"+customTemplate)

        character = WorldObject(
            self.templates,
            self.textGenerator,
            "character",
            objects={"name": tag},
            cfg=self.cfg,
            customTemplate=customTemplate
        )

        # print("GENERATED CHARACTER", character.__repr__())

        return character

    def transcriptToAnime(
        self,
        transcript,
        promptSuffix="",
        portrait_size=128,
        aggressiveMerging=False,
        savedcharacters=None,
        savedPortraits=None,
        savedVoices=None,
        savedGenders=None,
        actionDuration=5,
        settingDuration=2,
        imageFrequency=3,
        storyObjects=None,
        mainCharacterName=None,
    ):

        # make sure text generator is on cuda (can get out of sync if we ctrl+c during doGen() )
        # if self.textGenerator["name"].startswith("GPT3"):
        #    pass
        # else:
        #    self.textGenerator['pipeline'].model = self.textGenerator['pipeline'].model.cuda(
        #    )

        # extract characters
        if savedcharacters is None:
            _characters = {}
        else:
            _characters = savedcharacters
        if savedPortraits is None:
            portraits = {}
        else:
            portraits = savedPortraits
        if savedVoices is None:
            voices = {}
        else:
            voices = savedVoices
        if savedGenders is None:
            genders = {}
        else:
            genders = savedGenders

        tagBundles = self.getTagBundles(transcript)
        for line in transcript.split("\n"):
            tag = line.split(":")[0].strip().lower()
            if tag in ["setting", "action", "music", "sound effect"]:
                continue
            if aggressiveMerging:
                # tagn=self.normalizeTag(tag,tagBundles)
                tagn = self.mergeName(tag, _characters.keys())
            else:
                tagn = tag
            if tagn in _characters:
                continue
            else:
                character = self.generateNewCharacter(tag, _characters)

                print("GENERATED CHARACTER", character.__repr__())

                _characters[tagn] = character

        characters = list(_characters.values())

        # get voices

        for thisCharacter in characters:
            name = str(thisCharacter.getProperty("name"))
            gender = thisCharacter.getProperty("gender").lower()

            if name in voices:
                continue

            if self.usePITS:
                # voices[name]=random.randint(0,len(self.pitsTTS.hps.data.speakers)-1)
                if gender == "male":
                    voices[name] = random.choice(self.maleVoices)
                else:
                    voices[name] = random.choice(self.femaleVoices)
            else:
                if gender == "male":
                    voices[name] = random.choice(self.maleVoices)
                else:
                    voices[name] = random.choice(self.femaleVoices)
            genders[name] = gender
            description = thisCharacter.getProperty("description")

            print("GOT GENDER FOR:", name, "=", gender)

        # generate portraits

        for thisCharacter in characters:
            name = str(thisCharacter.getProperty("name"))

            if name in portraits:
                continue

            gender = thisCharacter.getProperty("gender")
            description = thisCharacter.getProperty("description")
            prompt = "close up headshot, high resolution color portrait of "+name+" "+gender+", "+description + \
                ", solid white background"

            # portrait = self.doGen(
            #    prompt, num_inference_steps=self.num_inference_steps)
            portrait = self.getPortrait(prompt, promptSuffix)

            portraits[name] = portrait
            yield {"debug": description}
            yield {"image": portrait,
                   "width": 1024,
                   "height": 1024,
                   }
            yield {"caption": "new character: %s: %s" % (name, description), "duration": settingDuration}

        lastPrompt = "an empty stage"
        t = 0
        settingImage = self.doGen(
            "an empty stage", num_inference_steps=self.num_inference_steps)

        for line in transcript.split("\n"):
            t += 1

            if len(line.split(":")) != 2:
                print("this should never happen!", line, transcript)
                continue

            tag = line.split(":")[0].strip().lower()
            description = line.split(":")[1].strip().lower()

            if imageFrequency is not None and t > imageFrequency and tag not in ["setting", "action"]:
                logging.info("creating extra image %s", tag)
                t = 0
                img = self.doGen(
                    lastPrompt, num_inference_steps=self.num_inference_steps)

                # generate video
                if self.video_mode:
                    video = generateGif(lastPrompt, img)
                else:
                    video = None

                if self.doImg2Img:
                    width, height = self.imageSizes[2], self.imageSizes[3]
                else:
                    width, height = self.imageSizes[2], self.imageSizes[3]

                yield {"image": img,
                       "width": width,
                       "height": height,
                       "video": video,
                       }
                settingImage = img

            if tag == "setting":

                prompt = description+promptSuffix

                prompt, didEnhance = self.enhancePrompt(
                    prompt, _characters, storyObjects)
                lastPrompt = prompt
                if didEnhance:
                    print("enhanced prompt", prompt)

                t = 0
                settingImage = self.doGen(
                    prompt, num_inference_steps=self.num_inference_steps)

                # generate video
                if self.video_mode:
                    video = generateGif(prompt, img)
                else:
                    video = None

                if self.doImg2Img:
                    width, height = self.imageSizes[2], self.imageSizes[3]
                else:
                    width, height = self.imageSizes[2], self.imageSizes[3]

                yield {"image": settingImage,
                       "width": width,
                       "height": height,
                       "video": video,
                       }
                yield {"caption": "Setting: %s" % description,
                       "duration": settingDuration}

            elif tag == "music":
                musicPrompt = description+self.musicSuffix
                audio = self.generate_track_by_prompt_vol(
                    musicPrompt, vol=0.25,
                    duration=self.musicDuration
                )
                yield {"music": audio}

            elif tag == "sound effect":

                # todo: implement
                audio, duration = self.generateAudio(
                    description, self.soundEffectDuration)

                yield {"sound effect": audio,
                       "description": description,
                       "duration": duration,
                       }

                # yield {"caption": "Sound Effect: %s" % description,
                #       "duration": settingDuration}

            elif tag == "action":

                prompt = description+promptSuffix
                lastPrompt = prompt
                prompt, didEnhance = self.enhancePrompt(
                    prompt, _characters, storyObjects)
                if didEnhance:
                    print("enhanced prompt", prompt)
                elif mainCharacterName is not None:
                    # add main character description
                    print("no character found, adding main character")
                    print("main character name", mainCharacterName)
                    print("characters", _characters.keys())
                    print("characters", _characters[mainCharacterName])
                    prompt += ", " + \
                        str(_characters[mainCharacterName].getProperty(
                            "description"))

                actionImage = self.doGen(
                    prompt, num_inference_steps=self.num_inference_steps)

                # generate video
                if self.video_mode:
                    video = generateGif(prompt, img)
                else:
                    video = None

                # for now this seems better
                t = 0
                settingImage = actionImage

                if self.doImg2Img:
                    width, height = self.imageSizes[2], self.imageSizes[3]
                else:
                    width, height = self.imageSizes[2], self.imageSizes[3]

                yield {"image": actionImage,
                       "width": width,
                       "height": height,
                       "video": video,
                       }

                yield {"caption": description,
                       "duration": actionDuration}

            else:
                print("Dying here?")
                if aggressiveMerging:
                    # tagn=self.normalizeTag(tag,tagBundles)
                    tagn = self.mergeName(tag, _characters.keys())
                else:
                    tagn = tag

                thisCharacter = _characters[tagn]

                name = str(thisCharacter.getProperty("name"))

                # thisImg = settingImage.copy()
                # name, dialogue = tagn,description
                dialogue = description
                voice = voices[name]
                gender = genders[name]
                portrait = portraits[name]
                # p2 = portrait.resize((portrait_size, portrait_size))
                # thisImg.paste(
                #    p2, (thisImg.size[0]-portrait_size, thisImg.size[1]-portrait_size))

                # print("about to die",dialogue, voice)

                if len(dialogue.strip()) == 0:
                    print("this should never happen!", transcript)
                    continue

                speech, duration = self.getTalkingHeadVideo(
                    portrait, dialogue, voice, gender, decimate=self.talking_head_decimate)

                yield {"speech": speech,
                       "duration": duration,
                       "name": name,
                       "dialogue": dialogue}

        return

    def openaiChatCompletion(
        self,
        model="gpt-3.5-turbo-instruct",
        messages=[],
        timeout=10,
        n=1,
        max_tokens=512
    ):

        # first combine all of the messages into a prompt
        prompt = ""
        for message in messages:
            prompt += message['role']+":\n "+message['content']+"\n"
            # prompt += message['content']+"\n"
        prompt += "assistant:\n"

        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            n=n,
            max_tokens=max_tokens,
        )

        return response

    # @retry(wait=wait_exponential(multiplier=1, min=4, max=10),before_sleep=before_sleep_log(logger, logging.WARNING))
    def createScreenplay(self, sceneDescription, previousMessages=[], n=1):

        systemprompt = self.screenplayPrompt

        messages = [
            {"role": "system", "content": systemprompt},
        ] + \
            previousMessages + \
            [
            {"role": "user", "content": sceneDescription},
            {"role": "user", "content": "SCREENPLAY:"},
        ]

        # print("Creating Screenplay", messages)

        # if self.use_gpt_for_chat_completion:
        if True:  # for now, we always just use GPT for chat completion
            # response = openai.ChatCompletion.create(
            if self.use_GPT4:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    timeout=10,
                    n=n,
                )


                output = []
                for choice in response.choices:
                    output += [choice.message.content]
                    #output += [choice.con]

            else:
                response = self.openaiChatCompletion(
                    model="gpt-3.5-turbo-instruct",
                    messages=messages,
                    timeout=10,
                    n=n,
                )

                output = []
                for choice in response.choices:
                    # output += [choice.message.content]
                    output += [choice.text]

        else:  # theoretically I should fix this so it doesn't break if textmodel is "GPT3"
            output = self.chatCompletion(
                messages, n=n, generation_prefix="setting:")

        return output

    def classify_text_openai0(self, text, categories=["setting", "action", "sound effect"]):
        prompt = f"Classify the following line of text into one of these categories: setting, action, or sound effect:\n\n{text}\n\nCategory:"

        response = openai.Completion.create(
            engine="text-curie-001",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.1,
        )

        response_text = response.choices[0].text.strip().lower()

        # Find the best matching category
        best_category = None
        best_match = 0
        for category in categories:
            match = response_text.count(category)
            if match > best_match:
                best_match = match
                best_category = category
        return best_category

    def classify_text_openai(self, text, categories=["setting", "action", "sound effect", "tone", "gesture"]):

        messages = example_classifications+[{"role": "user", "content": text}]

        if self.use_gpt_for_chat_completion:
            # response = openai.ChatCompletion.create(
            response = self.openaiChatCompletion(
                model="gpt-3.5-turbo-instruct",
                messages=messages, timeout=10
            )

            result = ''
            for choice in response.choices:
                # result += choice.message.content
                result += choice.text

            output = result.lower().strip()
        else:
            output = self.chatCompletion(messages)[-1]

        if result not in categories:
            print("THIS SHOULD NEVER HAPPEN", result)
            best_category = None
            best_match = 0
            for category in categories:
                match = result.count(category)
                if match > best_match:
                    best_match = match
                    best_category = category

            if best_match > 0:
                output = best_category
            else:
                output = "action"

            print("output", output)

        return output

    def validateScreenplay(self, screenplay):
        score = 0
        hasMusic = False
        out = []
        for line in screenplay.split("\n"):
            # skip blank lines
            if len(line.strip()) == 0:
                # score+=1 #don't penalize blank lines
                continue
            # skip lines w/out colons
            if ":" not in line:
                score += 2
                if self.fixAsides:
                    category = self.classify_text_openai(description)
                    line = category+": "+description
                else:
                    continue

            if len(line.split(":")) != 2:
                score += 1
                continue

            # tag cannot be empty
            if len(line.split(":")[0].strip()) == 0:
                score += 1
                continue
            # tag shouldn't be very long
            if len(line.split(":")[0].strip().split()) > 4:
                score += 1
                continue
            # line shouldn't be very long
            if len(line) > 240:
                score += 1
                continue
            # check for music
            tag = line.split(":")[0].strip().lower()
            description = line.split(":")[1].strip()
            if tag == "music":
                print("found music", line)
                hasMusic = True

            # check if "music" is in the tag, e.g. "final music:"
            if "music" in tag:
                category = "music"
                line = category + ": "+description
                hasMusic = True

            # fix some bad tags
            if tag == "sfx":
                if self.fixAsides:
                    category = self.classify_text_openai(description)
                else:
                    category = "sound effect"
                line = category+": "+description

            # fix some bad tags
            if tag == "sound effect":
                if self.fixAsides:
                    category = self.classify_text_openai(description)
                else:
                    category = "sound effect"
                line = category+": "+description

            if tag == "sound effects":
                if self.fixAsides:
                    category = self.classify_text_openai(description)
                else:
                    category = "sound effect"
                line = category+": "+description

            if "dialogue" in tag:
                if self.fixAsides:
                    category = self.classify_text_openai(description)
                else:
                    category = "action"
                line = category+": "+description

            if "supporting character" in tag:
                score += 1
                continue

            if tag == "gesture":
                tag = "action"

            if tag == "tone":
                # TODO: fix this
                continue

            # fix any tag that contains "setting", e.g "settings"
            if "setting" in tag:
                category = "setting"
                line = category+": "+description

            # some more tags like this
            tagsToFix = ["antagonist", "end", "flashback", "foreshadow", "prologue", "protagonist",
                         "start", "subplot", "theme", "title", "twist", "voiceover", "location"]
            for tagToFix in tagsToFix:
                if tag == tagToFix:
                    if self.fixAsides:
                        category = self.classify_text_openai(description)
                    else:
                        category = "action"
                    line = category+": "+description

            # description cannot be empty
            if len(line.split(":")[1].strip()) == 0:
                score += 1
                continue
            # remove ""s (but don't penalize score)
            line = re.sub("\"", "", line)

            # remove ()'s, *asides*, and [braces]
            patterns = [r'\((.*?)\)', r'\*(.*?)\*', r'\[(.*?)\]', r'\{(.*?)\}']
            for pattern in patterns:
                if re.search(pattern, line):
                    tag = re.sub(pattern, "", tag).strip()

                    if self.fixAsides:

                        for match in re.findall(pattern, line):
                            category = self.classify_text_openai(match)

                            if category == "gesture":
                                category = "action"

                            if category == "tone":
                                score += 0.5
                                line = re.sub(pattern, "", line)
                                continue

                            out += [category + ": " + tag + " " + match]

                    score += 1
                    line = re.sub(pattern, "", line)

            # remove []'s

            # remove ""'s
            if re.search("[^a-zA-Z0-9_.?!,';:\- ]", line):
                score += 1
            line = re.sub("[^a-zA-Z0-9_.?!,';:\- ]", "", line)
            if len(line.strip()) == 0:
                score += 1
                continue

            # reject if description contains no letters
            tag = line.split(":")[0].strip().lower()
            description = line.split(":")[1].strip()
            if re.search("[a-zA-Z]", description) == None:
                score += 1
                continue

            out += [line]

        # add music if there isn't any
        if hasMusic == False:
            out = ["music: %s" % self.riffusionSuffix]+out

        # print(out,hasMusic)

        return out, score

    def getValidScreenplay(self, sceneDescription, nTrials=3, previousMessages=[], allowed_errors=3, verbose=False):
        whichScreenplay = -1
        bestScreenplay = None
        _bestScreenplay = None
        bestScore = 999

        # first let's request all of the screenplays in parallel (this doesn't work because it forks the whole thing...)
        # pool = Pool(processes=4) #bad idea, forking this process is bad because of all the global variables
        # screenplays=pool.map(lambda x: self.createScreenplay(sceneDescription, previousMessages=previousMessages), range(nTrials))
        # createScreenplay

        if self.parallel_screenplays:
            screenplays = self.createScreenplay(
                sceneDescription, previousMessages=previousMessages, n=nTrials)

        for i in range(nTrials):
            try:
                print("CREATING SCREENPLAY, attempt", i, "of", nTrials)
                # s = self.createScreenplay(
                #    sceneDescription, previousMessages=previousMessages)

                if self.parallel_screenplays:
                    s = screenplays[i]
                else:
                    s = self.createScreenplay(
                        sceneDescription, previousMessages=previousMessages)[0]
                print(s)
            except Exception as e:
                print(e)
                traceback.print_exc()
                # print error
                print("FAILED TO CREATE SCREENPLAY, attempt", i, "of", nTrials)
                continue

            print("Validating SCREENPLAY, attempt", i, "of", nTrials)

            try:
                v, score = self.validateScreenplay(s)
            except Exception as e:
                print(e)
                traceback.print_exc()
                # print error
                print("FAILED TO VALIDATE SCREENPLAY, attempt", i, "of", nTrials)
                continue

            if verbose:
                print(s, score)

            # print the loglevel to see if it's working
            print("what happend to our log level?",
                  logging.getLogger().getEffectiveLevel())
            logging.info("screenplay:\n score %d/%d=%f",
                         score, len(v), score/len(v))

            if len(v) > 8 and score <= allowed_errors:

                if self.parallel_screenplays:
                    pass  # no point in returning early anymore, since we generate nTrials screenplays regardless
                else:
                    print("RETURNING EARLY", score, len(v), "\n".join(v))
                    return "\n".join(v)

            if len(v) > 8 and score/len(v) < bestScore:
                logging.info("new best score! %f", score/len(v))
                _bestScore = score
                bestScore = score/len(v)
                bestScreenplay = v
                _bestScreenplay = s
                whichScreenplay = i

        if bestScreenplay is None:
            print("unable to create screenplay!")
            s = self.createScreenplay(
                sceneDescription, previousMessages=previousMessages, n=1)[0]
            # print("GOT VALID SCREENPLAY",bestScore,len(v),"\n".join(v))
            v, score = self.validateScreenplay(s)
            return "\n".join(v)
        else:
            # print(_bestScreenplay, bestScore)
            print("Choose screenplay", whichScreenplay, "with score",
                  _bestScore, len(bestScreenplay), bestScore/len(bestScreenplay))
            return "\n".join(bestScreenplay)

    def createTranscriptGPT(self, novelSummary, characters, chapters, allScenes, whichChapter, whichScene, previousMessages=None, num_chapters=12, num_scenes=5, max_tokens=1000, additionalScenePrompt=None, conclusionPrompt=None, verbose=False):

        print("creating scene %d of chapter %d" % (whichScene, whichChapter))

        summarizeNovelMessage = str(WorldObject(
            self.templates,
            self.textGenerator,
            "explainNovelTemplate",
            objects={"novelSummary": novelSummary,
                     "novelCharacters": characters,
                     "novelChapters": chapters,
                     },
            cfg=self.cfg
        ))

        # remove lines that start with "<"
        summarizeNovelMessage = re.sub(r'\<.*?\>', '', summarizeNovelMessage)

        sceneSummary = allScenes[whichChapter -
                                 1].getProperty("scene %d summary" % whichScene)

        # print(summarizeNovelMessage)

        print(sceneSummary)

        if additionalScenePrompt:
            sceneSummary += additionalScenePrompt

        if conclusionPrompt is None:
            conclusionPrompt = " This is the last scene, so make sure to give the story a satisfying conclusion."

        if whichChapter == num_chapters and whichScene == num_scenes:
            sceneSummary += conclusionPrompt

        examplePrompt = exampleScreenplayPrompt.format(
            mainCharacter=characters.getProperty("main character name"),
            supportingCharacter1=characters.getProperty(
                "supporting character 1 name"),
            supportingCharacter2=characters.getProperty(
                "supporting character 2 name")
        )

        exampleTranscript = exampleScreenplayResult.format(MainCharacter=characters.getProperty("main character name"),
                                                           SupportingCharacter1=characters.getProperty(
            "supporting character 1 name"),
            SupportingCharacter2=characters.getProperty(
            "supporting character 2 name")
        )

        if previousMessages is None:


           # we should tell it in advance what scenes are in this chapter
            s = ""
            for i in range(1, num_scenes+1):
                s += "chapter {whichChapter} has the following scenes:\n\nscene {i} summary:\n{sceneSummary}\n".format(
                    whichChapter=whichChapter,
                    i=i,
                    sceneSummary=allScenes[whichChapter -
                                            1].getProperty("scene %d summary" % i)
                )
            chapter_scenes_message = {"role": "user", "content": s}


            messages = [
                {"role":"system","content":self.screenplayPrompt},
                {"role": "user", "content": summarizeNovelMessage},
                {"role": "user", "content": "Create a transcript for chapter 0, scene 1 with the following summary\n\n{sceneSummary}".format(
                    whichChapter=whichChapter, whichScene=whichScene, sceneSummary=examplePrompt)},
                # {"role": "user", "content": examplePrompt},
                # {"role": "user", "content": "SCREENPLAY:"},
                {"role": "assistant", "content": exampleTranscript},
                chapter_scenes_message,
                {"role": "user", "content": "Create a transcript for chapter {whichChapter}, scene {whichScene} with the following summary\n\n{sceneSummary}".format(
                    whichChapter=whichChapter, whichScene=whichScene, sceneSummary=sceneSummary)}
            ]

        else:

            if whichScene == 1:
                # we should tell it in advance what scenes are in this chapter
                s = ""
                for i in range(1, num_scenes+1):
                    s += "chapter {whichChapter} has the following scenes:\n\nscene {i} summary:\n{sceneSummary}\n".format(
                        whichChapter=whichChapter,
                        i=i,
                        sceneSummary=allScenes[whichChapter -
                                               1].getProperty("scene %d summary" % i)
                    )

                previousMessages = previousMessages+[{"role": "user", "content": s}]

                print("added chapter scenes")
            else:
                print("skipping chapter scenes", whichScene)

            messages = previousMessages+[
                {"role": "user", "content": "Create a transcript for chapter {whichChapter}, scene {whichScene} with the following summary\n\n{sceneSummary}".format(
                    whichChapter=whichChapter, whichScene=whichScene, sceneSummary=sceneSummary)}
            ]

        logging.info("Creating scene with description: %s", sceneSummary)

        print("MESSAGES", messages)

        # response=animeBuilder.createScreenplay(sceneSummary,messages)
        response = self.getValidScreenplay(
            sceneSummary, previousMessages=messages)
        # response=animeBuilder.getValidScreenplay(sceneSummary)

        outputMessages = messages+[
            {"role": "user", "content": sceneSummary},
            {"role": "user", "content": "SCREENPLAY:"},
            {"role": "assistant", "content": response},
        ]

        return response, outputMessages

    def createChaptersFromWorldObject(self, novelSummary,
                                      characters,
                                      k=3):

        # first format our chapter prompt
        emptyChapterTemplate = "\n".join(["""chapter {i} title:
<chapter {i} title>
chapter {i} summary:
<chapter {i} title>""".format(i=i) for i in range(1, k+1)])
        formattedSystemPrompt = self.chapterPrompt.format(
            k=k, emptyChapterTemplate=emptyChapterTemplate)

        # first we need to build a custom template for the novel
        customTemplate = """
{formattedSystemPrompt}

summary: 
{examplechapterPrompt}

{exampleChapterResult}

characters:
{novelCharacters}

summary: 
{thisNovelSummary}

Remember, chapter summary should be a brief sentence or two describing what happens in the chapter.

""".format(
            formattedSystemPrompt=formattedSystemPrompt,
            thisNovelSummary=novelSummary.getProperty("summary"),
            examplechapterPrompt=examplechapterPrompt,
            exampleChapterResult=exampleChapterResults[k],
            novelCharacters=str(characters).split("\n", 1)[1]
        )

        emptyChapterTemplates = ["""chapter <i> title:
{chapter <i> title:TEXT:}
chapter <i> summary:
{chapter <i> summary:TEXT:}""".replace("<i>", str(i)) for i in range(1, k+1)]

        # add a comment at the start of final chapter
        emptyChapterTemplates[-1] = ">this is the final chapter\n" + \
            emptyChapterTemplates[-1]

        emptyChapterTemplate = "\n".join(emptyChapterTemplates)

        customTemplate += "\n"+emptyChapterTemplate

        print("how's it going?", customTemplate)

        # now we need to build a world object for the novel
        w = WorldObject(
            self.templates,
            self.textGenerator,
            "novelChapters",
            customTemplate=customTemplate,
            cfg=self.cfg)

        print("\n===\nhow did it go?\n===\n", w.filledTemplate)

        output = str(w).split("\n", 1)[1]

        return output

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def novelToChapters(self, novelSummary, novelCharacters, previousMessages=None, k=12):

        emptyChapterTemplate = "\n".join(["""chapter {i} title:
<chapter {i} title>
chapter {i} summary:
<chapter {i} title>""".format(i=i) for i in range(1, k+1)])

        systemPrompt = self.chapterPrompt.format(
            k=k, emptyChapterTemplate=emptyChapterTemplate)

        if previousMessages is None:
            previousMessages = [
                {"role": "user", "content": examplechapterPrompt},
                {"role": "user", "content": emptyChapterTemplate},
                {"role": "assistant", "content": exampleChapterResults[k]},
            ]

        messages = [
            {"role": "system", "content": systemPrompt},
        ] + \
            previousMessages + \
            [
            {"role": "user", "content": str(novelCharacters)},
            {"role": "user", "content": str(novelSummary)},
            {"role": "user", "content": emptyChapterTemplate}
        ]

        # print("here3",messages)
        # logging.info(messages)

        if self.use_gpt_for_chat_completion:
            # response = openai.ChatCompletion.create(
            response = self.openaiChatCompletion(
                model="gpt-3.5-turbo-instruct",
                messages=messages, timeout=10
            )

            result = ''
            for choice in response.choices:
                # result += choice.message.content
                result += choice.text
        else:
            result = self.chatCompletion(messages)[0]

        return result

    def validateChapters(self, novelChapters, k=12, verbose=False):

        # if any lines contain a ":", then split on the ":" and move the 2nd half to the next line
        # this is to fix a bug in the GPT-3 engine where it sometimes puts a ":" in the middle of a line
        newLines = []
        for line in novelChapters.split('\n'):
            if ":" in line and not line.endswith(":"):
                parts = line.split(":")
                newLines.append(parts[0]+":")
                newLines.append(parts[1])
            else:
                newLines.append(line)
        novelChapters = '\n'.join(newLines)

        # remove blank lines
        customTemplate = ""
        for line in novelChapters.split("\n"):
            # drop blank lines
            if len(line.strip()) == 0:
                continue
            line = line.strip()
            # tags should be lowercase
            if line[-1] == ":":
                line = line.lower()
            customTemplate += line+"\n"

        if verbose:
            print(customTemplate)

        w = WorldObject(
            self.templates,
            self.textGenerator,
            "novelChapters",
            customTemplate=customTemplate,
            cfg=self.cfg)

        score = 0
        for i in range(1, k+1):
            if w.has("chapter %d title" % i) and w.has("chapter %d summary" % i):
                score += 1

        logging.info("%s \n score %d", novelChapters, score)

        return w, score

    def getValidChapters(self, novelSummary, characters, k=12, nTrials=3, verbose=False):

        bestNovel = None
        bestScore = 0
        for i in range(nTrials):
            c = self.novelToChapters(novelSummary, characters, k=k)
            w, score = self.validateChapters(c, k=k)
            if score == k:
                return w
            if score > bestScore:
                bestNovel = w
                bestScore = score
        print("failed to generate novel", score)
        return w

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def chapterToScenes(self,
                        novelSummary,
                        characters,
                        chapters,
                        chapterTitle,
                        chapterSummary,
                        whichChapter,
                        previousMessages=None,
                        k=5,
                        numChapters=12
                        ):
        emptyScenesTemplate = "\n".join(["""scene {i} summary:
<scene {i} summary>""".format(i=i)
            for i in range(1, k+1)
        ])

        systemPrompt = self.scenePrompt.format(
            numScenes=k, emptyScenesTemplate=emptyScenesTemplate)

        if previousMessages is None:
            messages = [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": exampleScenesPrompt},
                {"role": "assistant", "content": exampleScenesResult[k]},
                {"role": "user", "content": str(characters)},
                {"role": "user", "content": str(novelSummary)},
                # {"role": "user", "content": str(chapters)},
                {"role": "user", "content": "generate scenes for chapter %d of this novel which has a total of %d chapters" %
                    (whichChapter, numChapters)},
                {"role": "user", "content": str(chapterTitle)},
                {"role": "user", "content": str(chapterSummary)},
                {"role": "user", "content": emptyScenesTemplate},
            ]
        else:
            messages = previousMessages+[
                {"role": "user", "content": "generate scenes for chapter %d of this novel which has a total of %d chapters" %
                    (whichChapter, numChapters)},
                {"role": "user", "content": str(chapterTitle)},
                {"role": "user", "content": str(chapterSummary)},
                {"role": "user", "content": emptyScenesTemplate},
            ]

        if self.use_gpt_for_chat_completion:
            # response = openai.ChatCompletion.create(
            response = self.openaiChatCompletion(
                model="gpt-3.5-turbo-instruct",
                messages=messages,
                timeout=10,
            )

            result = ''
            for choice in response.choices:
                # result += choice.message.content
                result += choice.text
        else:
            result = self.chatCompletion(messages)[0]

        outputMessages = messages+[{"role": "assistant", "content": result}]

        return result, outputMessages

    def validateScenes(self, chapterScenes, k=5, verbose=False):

        # if any lines contain a ":", then split on the ":" and move the 2nd half to the next line
        # this is to fix a bug in the GPT-3 engine where it sometimes puts a ":" in the middle of a line
        newLines = []
        for line in chapterScenes.split('\n'):
            if ":" in line and not line.endswith(":"):
                parts = line.split(":")
                newLines.append(parts[0]+":")
                newLines.append(parts[1])
            else:
                newLines.append(line)
        chapterScenes = '\n'.join(newLines)

        # remove blank lines
        customTemplate = ""
        for line in chapterScenes.split("\n"):
            # drop blank lines
            if len(line.strip()) == 0:
                continue
            line = line.strip()
            # tags should be lowercase
            if line[-1] == ":":
                line = line.lower()
            customTemplate += line+"\n"

        logging.info(customTemplate)

        w = WorldObject(
            self.templates,
            self.textGenerator,
            "chapterScenes",
            customTemplate=customTemplate,
            cfg=self.cfg)

        score = 0
        k = 1
        while w.has("scene %d summary" % k):
            score += 1
            k += 1

        return w, score

    def getValidScenes(
        self,
        novelSummary,
        characters,
        chapters,
        chapterTitle,
        chapterSummary,
        whichChapter,
        k=5,
        nTrials=3,
        previousMessages=None,
        numChapters=12,
        verbose=False
    ):

        # first try using GPT
        if self.use_gpt_for_chat_completion:
            bestNovel = None
            bestScore = -999
            bestMessages = None
            for i in range(nTrials):
                c, messages = self.chapterToScenes(novelSummary,
                                                   characters,
                                                   chapters,
                                                   chapterTitle,
                                                   chapterSummary,
                                                   whichChapter,
                                                   previousMessages=previousMessages,
                                                   numChapters=numChapters,
                                                   k=k
                                                   )

                w, foundScenes = self.validateScenes(c, k=k)

                logging.info("FoundScenes %d / %d", foundScenes, k)

                if foundScenes == k:
                    return w, messages
                if foundScenes > k:
                    score = k-foundScenes
                    print("too many scenes!", score)
                    if score > bestScore:
                        bestNovel = w
                        bestScore = score
                        bestMessages = messages
            print("failed to generate novel", foundScenes)

        if previousMessages is None:
            previousMessages = []

        fallbackTemplate = ""

        # add scene prompt
        emptyScenesTemplate = "\n".join(["""scene {i} summary:
<scene {i} summary>""".format(i=i)
            for i in range(1, k+1)
        ])

        systemPrompt = self.scenePrompt.format(
            numScenes=k, emptyScenesTemplate=emptyScenesTemplate)

        fallbackTemplate += systemPrompt+"\n"

        if len(previousMessages) == 0:
            fallbackTemplate += "chapter summary:\n"+exampleScenesPrompt+"\n"
            fallbackTemplate += exampleScenesResult[k]

        # combine chapters
        chapterString = ""
        for i in range(1, numChapters+1):
            chapterString += "chapter %d title:\n%s\nchapter %d summary:\n%s\n" % (
                i, chapters.getProperty("chapter %d title" % i), i, chapters.getProperty("chapter %d summary" % i))

        # make sure to include novel summary
        fallbackTemplate += """

We will now be generating scenes for the following novel:

{chapterString}

{novelCharacters}

novel title:
{novelTitle}
novel summary:
{novelSummary}

""".format(
            k=k,
            novelTitle=novelSummary.getProperty("title"),
            novelSummary=novelSummary.getProperty("summary"),
            novelCharacters=str(characters).split("\n", 1)[1],
            chapterString=chapterString
        )

        for message in previousMessages:
            if message["role"] == "user":
                fallbackTemplate += message["content"]+"\n\n"
            if message["role"] == "assistant":
                fallbackTemplate += message["content"]+"\n\n"

        emptyScenesTemplate = "\n".join(["""scene <i> summary:
{scene <i> summary:TEXT:}""".replace("<i>", str(i))
            for i in range(1, k+1)
        ])

        fallbackTemplate += "> generate scenes for chapter %d of this novel\n" % whichChapter
        fallbackTemplate += self.bonusSceneInstruction

        fallbackTemplate += "chapter title:\n"+chapterTitle+"\n\n"

        fallbackTemplate += "chapter summary:\n" + \
            chapterSummary+"\n\n"+emptyScenesTemplate

        print("==========\n\nChapter "+str(whichChapter) +
              "fallback template\n====\n", fallbackTemplate, "\n\n")

        w = WorldObject(
            self.templates,
            self.textGenerator,
            "chapterToScenes",
            customTemplate=fallbackTemplate,
            cfg=self.cfg
        )

        # print("\n====\nproduced object:\n"+str(w)+"\n\n")

        # print("with filled template:\n"+w.filledTemplate+"\n\n")

        messages = previousMessages+[
            {"role": "user", "content": "generate scenes for chapter %d of this novel" %
             whichChapter},
            {"role": "user", "content": "chapter title:\n"+str(chapterTitle)},
            {"role": "user", "content": "chapter summary:\n" +
                str(chapterSummary)},
            # {"role": "assistant", "content": str(w).split("\n", 1)[1]}
        ]

        # chapter_scenes = str(w).split("\n", 1)[1]
        chapter_scenes = str(w)

        # print("GIR",messages)

        print("generated fallback scenes", chapter_scenes)

        output_messages = messages+[
            {"role": "assistant", "content": str(w).split("\n", 1)[1]}
        ]

        return chapter_scenes, output_messages

        # return bestNovel, bestMessages

    def chaptersToScenes(
        self,
        novelSummary,
        characters,
        chapters,
        numChapters=12,
        numScenes=5,
        nTrials=3
    ):

        output = []

        previousMessages = None
        for whichChapter in range(1, numChapters+1):

            chapterTitle = chapters.getProperty(
                "chapter %d title" % (whichChapter))
            chapterSummary = chapters.getProperty(
                "chapter %d summary" % (whichChapter))

            if previousMessages is not None and len(previousMessages) > 20:
                previousMessages = previousMessages[:5]+previousMessages[-15:]

            c, messages = self.getValidScenes(
                novelSummary,
                characters,
                chapters,
                chapterTitle,
                chapterSummary,
                whichChapter=whichChapter,
                k=numScenes,
                nTrials=nTrials,
                previousMessages=previousMessages,
                numChapters=numChapters
            )

            print("\n\nchapter", whichChapter, chapterTitle, chapterSummary)
            print(c)

            output += [c]

            previousMessages = messages

            # print("What??", len(previousMessages), previousMessages)

        return output

    # these methods are special because they take text templates as inputs instead of
    # WorldObjects

    def create_novel_summary(self, story_objects):

        story_objects = story_objects.replace("\r\n", "\n")

        storyObjects = WorldObject(
            self.templates,
            self.textGenerator,
            "storyObjects",
            customTemplate=story_objects,
            cfg=self.cfg
        )

        '''
        novelSummary = WorldObject(
            self.templates,
            self.textGenerator,
            "novelSummary",
            # objects={"title":"The big one",
            #        "summary":"A group of sexy female lifeguards take up surfing"}
            objects={"storyObjects": storyObjects}
        )

        novel_summary = str(novelSummary)
        return novel_summary.split('\n', 1)[1]

        '''

        if storyObjects.has("novel suggestion"):
            novelSuggestion = storyObjects.getProperty("novel suggestion")
            logging.info("novel suggestion: %s", novelSuggestion)
        else:
            novelSuggestion = None

        if storyObjects.has("character type"):
            if novelSuggestion:
                novelSuggestion += "\ncharacter type = %s \n" % storyObjects.getProperty(
                    "character type")
            else:
                novelSuggestion = "\ncharacter type = %s \n" % storyObjects.getProperty(
                    "character type")

        logging.info("here %s", novelSuggestion)

        if self.use_gpt_for_chat_completion:
            novel_summary = self.chatGPTFillTemplate2(
                self.templates["novelSummary"], "novelSummary", extraInfo=novelSuggestion)
        else:

            nso = {"storyObjects": storyObjects}
            if storyObjects.has("character type"):
                nso["character type"] = storyObjects.getProperty(
                    "character type")

            if storyObjects.has("novel suggestion"):
                novelSuggestion = storyObjects.getProperty("novel suggestion")
                nso["novel suggestion"] = novelSuggestion

            novelSummary = WorldObject(
                self.templates,
                self.textGenerator,
                "novelSummary",
                # objects={"title":"The big one",
                #        "summary":"A group of sexy female lifeguards take up surfing"}
                objects=nso,
                cfg=self.cfg
            )
            novel_summary = str(novelSummary)
            novel_summary = novel_summary.split('\n', 1)[1]

        if novel_summary is None:
            novelSummary = WorldObject(
                self.templates,
                self.textGenerator,
                "novelSummary",
                # objects={"title":"The big one",
                #        "summary":"A group of sexy female lifeguards take up surfing"}
                objects={"storyObjects": storyObjects},
                cfg=self.cfg
            )

            novel_summary = str(novelSummary)
            novel_summary = novel_summary.split('\n', 1)[1]

        story_objects_out = str(storyObjects).split("\n", 1)[1]

        # print("about to die", story_objects,
        #      storyObjects.filledTemplate, story_objects_out)

        return {'story_objects': story_objects_out, 'novel_summary': novel_summary}

    def create_characters(self, story_objects, novel_summary):

        storyObjects = WorldObject(
            self.templates,
            self.textGenerator,
            "storyObjects",
            customTemplate=story_objects,
            cfg=self.cfg
        )

        novelSummary = WorldObject(
            self.templates,
            self.textGenerator,
            "novelSummary",
            customTemplate=novel_summary,
            cfg=self.cfg
        )

        objects = {"novelSummary": novelSummary}

        if self.use_gpt_for_chat_completion:
            novel_characters = self.chatGPTFillTemplate2(
                templates["novelCharacters"], "novelCharacters", objects=objects)
        else:
            characters = WorldObject(
                self.templates,
                self.textGenerator,
                "novelCharacters",
                objects={"novelSummary": novelSummary,
                         "storyObjects": storyObjects
                         },
                cfg=self.cfg
            )
            novel_characters = str(characters).split('\n', 1)[1]

        if novel_characters is not None:
            return novel_characters

        characters = WorldObject(
            self.templates,
            self.textGenerator,
            "novelCharacters",
            objects={"novelSummary": novelSummary,
                     "storyObjects": storyObjects
                     },
            cfg=self.cfg
        )

        return str(characters).split('\n', 1)[1]
        """
        
        """

    def create_chapters(self, story_objects, novel_summary, _characters, num_chapters, nTrials=3):
        storyObjects = WorldObject(
            self.templates,
            self.textGenerator,
            "storyObjects",
            customTemplate=story_objects,
            cfg=self.cfg
        )

        novelSummary = WorldObject(
            self.templates,
            self.textGenerator,
            "novelSummary",
            customTemplate=novel_summary,
            cfg=self.cfg
        )

        characters = WorldObject(
            self.templates,
            self.textGenerator,
            "novelCharacters",
            customTemplate=_characters,
            cfg=self.cfg
        )

        if self.use_gpt_for_chat_completion:

            chapters = self.getValidChapters(
                novelSummary,
                characters,
                k=num_chapters,
                nTrials=nTrials
            )

            output = str(chapters).split('\n', 1)[1]

        else:
            output = self.createChaptersFromWorldObject(
                novelSummary,
                characters,
                k=num_chapters,
            )

        return output

    def create_scenes(self, story_objects, novel_summary, _characters, _chapters, num_chapters, num_scenes, nTrials=3):

        novelSummary = WorldObject(
            self.templates,
            self.textGenerator,
            "novelSummary",
            customTemplate=novel_summary,
            cfg=self.cfg
        )

        characters = WorldObject(
            self.templates,
            self.textGenerator,
            "novelCharacters",
            customTemplate=_characters,
            cfg=self.cfg
        )

        chapters = WorldObject(
            self.templates,
            self.textGenerator,
            "chapters",
            customTemplate=_chapters,
            cfg=self.cfg
        )

        scenes = self.chaptersToScenes(
            novelSummary,
            characters,
            chapters,
            numChapters=num_chapters,
            numScenes=num_scenes,
            nTrials=nTrials
        )

        return "\n===\n".join([str(x).split('\n', 1)[1] for x in scenes])

    def generate_movie_data(self, story_objects, novel_summary, _characters, _chapters, scenes, num_chapters, num_scenes, aggressive_merging=True,
                            portrait_size=128, startChapter=None, startScene=None,skipGeneration=False):
        # Process the inputs and generate the movie data
        # This is where you would include your existing code to generate the movie elements
        # For demonstration purposes, we'll just yield some dummy elements

        print("creating movie")

        if startChapter is None:
            startChapter = 1
        if startScene is None:
            startScene = 1

        storyObjects = WorldObject(
            self.templates,
            self.textGenerator,
            "storyObjects",
            customTemplate=story_objects,
            cfg=self.cfg
        )

        if storyObjects.has("scene prompt"):
            additionalScenePrompt = storyObjects.getProperty("scene prompt")
        else:
            additionalScenePrompt = None

        if storyObjects.has("conclusion prompt"):
            conclusionPrompt = storyObjects.getProperty("conclusion prompt")
        else:
            conclusionPrompt = None

        # convert back into correct format
        novelSummary = WorldObject(
            self.templates,
            self.textGenerator,
            "novelSummary",
            customTemplate=novel_summary,
            cfg=self.cfg
        )

        characters = WorldObject(
            self.templates,
            self.textGenerator,
            "novelCharacters",
            customTemplate=_characters,
            cfg=self.cfg
        )

        chapters = WorldObject(
            self.templates,
            self.textGenerator,
            "chapters",
            customTemplate=_chapters
        )

        all_scenes = scenes.split("===")

        allScenes = [
            WorldObject(
                self.templates,
                self.textGenerator,
                "chapterScenes",
                customTemplate=_scenes,
                cfg=self.cfg
            )

            for _scenes in all_scenes
        ]

        print("generating characters")

        mainCharacter = WorldObject(
            self.templates,
            self.textGenerator,
            "character",
            objects={
                "name": characters.getProperty("main character name"),
                "description text": characters.getProperty("main character description"),
            },
            cfg=self.cfg
            # verbose=True
        )

        supportingCharacter1 = WorldObject(
            self.templates,
            self.textGenerator,
            "character",
            objects={
                "name": characters.getProperty("supporting character 1 name"),
                "description text": characters.getProperty("supporting character 1 description"),
            },
            cfg=self.cfg
        )

        supportingCharacter2 = WorldObject(
            self.templates,
            self.textGenerator,
            "character",
            objects={
                "name": characters.getProperty("supporting character 2 name"),
                "description text": characters.getProperty("supporting character 2 description"),
            },
            cfg=self.cfg
        )

        antagonist = WorldObject(
            self.templates,
            self.textGenerator,
            "character",
            objects={
                "name": characters.getProperty("antagonist name"),
                "description text": characters.getProperty("antagonist description"),
            },
            cfg=self.cfg
            # verbose=True
        )

        savedcharacters = {
            str(mainCharacter.getProperty("name").lower()): mainCharacter,
            str(supportingCharacter1.getProperty("name").lower()): supportingCharacter1,
            str(supportingCharacter2.getProperty("name").lower()): supportingCharacter2,
            str(antagonist.getProperty("name").lower()): antagonist,
        }
        savedPortraits = {}
        savedVoices = {}
        savedGenders = {}

        previousScene = None
        previousMessages = None

        yield {"debug": "new movie",
               "title": novelSummary.getProperty("title"),
               "summary": novelSummary.getProperty("summary"),
               "story_objects": story_objects,
               "novel_summary": novel_summary,
               "characters": _characters,
               "chapters": _chapters,
               "scenes": scenes,
               "num_chapters": num_chapters,
               "num_scenes": num_scenes,
               }

        print("starting movie")

        # make some music
        musicPrompt = self.musicSuffix

        audio = self.generate_track_by_prompt_vol(
            musicPrompt, vol=0.25,
            duration=self.musicDuration
        )

        yield {"music": audio}

        for whichChapter in range(1, num_chapters+1):
            for whichScene in range(1, num_scenes+1):

                # skip to the desired scene
                if whichChapter < startChapter or (whichChapter == startChapter and whichScene < startScene):
                    continue

                yield {"debug": "new scene",
                       "chapter": whichChapter,
                       "scene": whichScene}

                if previousScene is not None:
                    previousMessages = previousScene[1]

                    # trim messages when n>3 1+3*n=10

                    if len(previousMessages) > self.max_previous_scenes*3:
                        previousMessages = previousMessages[:3] + \
                            previousMessages[-9:]

                    thisScene = self.createTranscriptGPT(
                        novelSummary,
                        characters,
                        chapters,
                        allScenes,
                        whichChapter,
                        whichScene,
                        previousMessages,
                        num_chapters=num_chapters,
                        num_scenes=num_scenes,
                        additionalScenePrompt=additionalScenePrompt,
                        conclusionPrompt=conclusionPrompt
                    )

                else:

                    thisScene = self.createTranscriptGPT(
                        novelSummary,
                        characters,
                        chapters,
                        allScenes,
                        whichChapter,
                        whichScene,
                        num_chapters=num_chapters,
                        num_scenes=num_scenes,
                        additionalScenePrompt=additionalScenePrompt,
                        conclusionPrompt=conclusionPrompt
                    )

                s = thisScene[0]

                if previousMessages:
                    print("what??", len(previousMessages))

                yield {"debug": "transcript",
                       "whichChapter": whichChapter,
                       "whichScene": whichScene,
                       "transcript": s,
                       }

                if False and novelSummary.has("characterType"):
                    promptSuffix = ", " + \
                        novelSummary.getProperty("characterType")+self.suffix
                else:
                    promptSuffix = self.suffix

                if storyObjects.has("prompt suffix"):
                    promptSuffix = ", " + \
                        storyObjects.getProperty("prompt suffix")+self.suffix
                    

                if skipGeneration == False:

                    anime = self.transcriptToAnime(
                        s,
                        portrait_size=portrait_size,
                        promptSuffix=promptSuffix,
                        savedcharacters=savedcharacters,
                        savedPortraits=savedPortraits,
                        savedVoices=savedVoices,
                        savedGenders=savedGenders,
                        aggressiveMerging=aggressive_merging,
                        storyObjects=storyObjects,
                        mainCharacterName=str(
                            mainCharacter.getProperty("name").lower()),
                    )
                    for storyElement in anime:
                        yield storyElement

                previousScene = thisScene

        print("\n\n\nMOVIE DONE!\n\n\n")
        yield {"debug": "movie completed successfully"}
        yield {"caption": "THE END",
               "duration": 1}

    def validate(self, result, keys, templateName):
        # Remove blank lines from the result
        result = '\n'.join(line for line in result.split('\n') if line.strip())

        # if any lines contain a ":", then split on the ":" and move the 2nd half to the next line
        # this is to fix a bug in the GPT-3 engine where it sometimes puts a ":" in the middle of a line
        newLines = []
        for line in result.split('\n'):
            if ":" in line and not line.endswith(":"):
                parts = line.split(":")
                newLines.append(parts[0]+":")
                newLines.append(parts[1])
            else:
                newLines.append(line)
        result = '\n'.join(newLines)

        print("result", result)

        # fix any keys that are correct (expect for the wrong case) by substituting the correct case
        for key in keys:

            # match the key ignoring case
            m = re.search(r'(?i)'+key+r':', result)
            if m:
                # replace the key with the correct case
                result = result.replace(m.group(0), key+':')

        w = WorldObject(
            self.templates,
            self.textGenerator,
            templateName,
            customTemplate=result,
            cfg=self.cfg
        )
        for key in keys:
            if not w.has(key):
                print("missing key", key)
                return False, None
            p = w.getProperty(key)
            if re.search(r'[:\[\]{}<>]', p):
                return False, None
        return True, result

    # @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def chatGPTFillTemplate(animeBuilder, template, templateName, exampleTemplate=None, extraInfo=None, nTrials=3):

        templateSystemPrompt = """
You are TemplateEngine an advanced software system that takes a template and fills it with appropriate text

The template looks like:

{exampleTemplate}

The content in <>'s replaced with appropriate text

the text subsituted for <>'s NEVER contains ":"s
the text subsituted for <>'s is ALWAYS a single line
the text subsituted for <>'s ALWAYS appears on its own line

The system ALWAYS returns a filled template matching the format of the user input
The resulting 

The system NEVER refers to itself as an AI 
the system NEVER refuses to produce a list of scenes, 
the system NEVER does anything other than produce a formatted template matching the user's description
the system NEVER refers to itself as "the system"
the system NEVER says it cannot generate a list of scenes

the system NEVER uses ""s ()'s {{}}'s []'s or nonstandard punctuation  

the user MAY send the same template multiple times.  In this case, the system will REDO
the template with different plausible values

the system NEVER says "I'm sorry, it seems like you copied the previous input without changing it."
or anything similar

the system NEVER complains about missing keys, it just happily ignores them

The system never says "I'm sorry, but I cannot generate inappropriate or offensive content."



"""

        # replace \r\n everywhere with \n
        template = template.replace("\r\n", "\n")

        pattern = r'\{([^:{}]+)(:[^:]*:.*?)?\}'
        t = re.sub(pattern, r'<\1>', template)
        tt = [x.strip() for x in t.split("\n\n") if len(x.strip()) > 0]
        if exampleTemplate is None:
            exampleTemplate = tt[-1]
        formattedSystemPrompt = templateSystemPrompt.format(
            exampleTemplate=exampleTemplate)

        # logging.info("system prompt:\n%s",formattedSystemPrompt)
        if extraInfo is not None:
            logging.info("extra info:\n%s", extraInfo)

        messages = [
            {"role": "system", "content": formattedSystemPrompt}
        ]
        for example in tt[:-1]:
            messages += [{"role": "user", "content": exampleTemplate},
                         {"role": "assistant", "content": example}
                         ]
        if extraInfo:
            messages += [{"role": "user", "content": extraInfo}]
        messages += [{"role": "user", "content": tt[-1]}]
        keys = [line.split(":")[0]
                for line in tt[-1].split("\n") if ":" in line]

        # print("MESSAGES", messages )

        for i in range(nTrials):
            if animeBuilder.use_gpt_for_chat_completion:
                # response = openai.ChatCompletion.create(
                response = animeBuilder.openaiChatCompletion(
                    model="gpt-3.5-turbo-instruct",
                    messages=messages,
                    timeout=10
                )

                # result = response.choices[0].message.content
                result = response.choices[0].text
            else:
                result = animeBuilder.chatCompletion(messages)[0]

            logging.info("RESPONSE %d %s", i, result)
            isValid, result = animeBuilder.validate(
                result, keys, "novelSummary")
            if isValid:
                return result
        print("this should never happen!")
        # return random.choice(tt[:-1])
        return None

    def chatGPTFillTemplate2(animeBuilder, template, templateName, extraInfo=None, objects=None, nTrials=3):

        # replace \r\n everywhere with \n
        template = template.replace("\r\n", "\n")

        pattern = r'\{([^:{}]+)(:[^:]*:.*?)?\}'
        t = re.sub(pattern, r'<\1>', template)
        tt = [x.strip() for x in t.split("\n\n") if len(x.strip()) > 0]

        exampleTemplate = tt[-1]

        _extraInfo = []

        # if objects is not None:
        if True:
            # first fill in all of the values from objects
            def get_object_property(object_name, property_name):
                obj = objects[object_name]

                if obj and obj.has(property_name):
                    return obj.getProperty(property_name)
                else:
                    return f"{{{object_name}.{property_name}}}"

            def createWorldObject(property_type, overrides=None):
                if overrides is not None:
                    # TODO:fixme
                    objects = {}
                else:
                    objects = {}
                w = WorldObject(
                    animeBuilder.templates,
                    animeBuilder.textGenerator,
                    property_type,
                    objects=objects,
                    cfg=animeBuilder.cfg)

                return str(w)

            def replacement_function(match_obj):

                print("GOT HERE", match_obj)

                matched_text = match_obj.group(1)
                match_split = matched_text.split(':')

                if len(match_split) >= 2:
                    property_name, property_type = match_split[:2]
                    overrides = match_split[2] if len(
                        match_split) == 3 else None

                    if property_type != "TEXT":
                        s = createWorldObject(property_type, overrides)

                        line = f"{{{matched_text}}}"+"="+s
                        pattern = r'\{([^:{}]+)(:[^:]*:.*?)?\}'
                        line = re.sub(pattern, r'<\1>', line)

                        # _extraInfo.append(f"{{{matched_text}}}"+"="+s)
                        _extraInfo.append(line)

                        print("RETURNING HERE", _extraInfo, s)

                        # return f"{{{matched_text}}}"
                        return s
                    else:
                        return f"{{{matched_text}}}"
                else:
                    property_split = matched_text.split('.')
                    if len(property_split) == 2:
                        object_name, property_name = property_split
                        return get_object_property(object_name, property_name)
                    else:
                        return f"{{{matched_text}}}"

            pattern = r'\{([^}]+)\}'
            augmentedTemplate = re.sub(pattern, replacement_function, template)
        else:
            augmentedTemplate = template

        logger.info("augmentedTemplate %s", augmentedTemplate)

        if extraInfo is None:
            extraInfo = ""

        # logging.info("_extraInfo %s",_extraInfo)

        for line in _extraInfo:
            extraInfo += line+"\n"

        extraInfo_lines = []
        filteredTemplate_lines = []

        for line in augmentedTemplate.split('\n'):
            if line.startswith('>'):
                # Remove '>' and add line to extraInfo_lines
                extraInfo_lines.append(line[1:].strip())
            else:
                filteredTemplate_lines.append(line)

        extraInfo = '\n'.join(extraInfo_lines) + '\n' + extraInfo
        filteredTemplate = '\n'.join(filteredTemplate_lines)

        if len(extraInfo) == 0:
            extraInfo = None

        # print("about to die\n==\n", extraInfo,
        #      "\n==\n", filteredTemplate, "\n==")

        return animeBuilder.chatGPTFillTemplate(filteredTemplate, templateName, exampleTemplate=exampleTemplate, extraInfo=extraInfo, nTrials=nTrials)
