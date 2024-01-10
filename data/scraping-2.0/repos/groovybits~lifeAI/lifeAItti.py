#!/usr/bin/env python

## Life AI Stable Diffusion module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import io

from diffusers import StableDiffusionPipeline
import torch
from transformers import logging as trlogging
import re
import logging
import time
from openai import OpenAI
import base64
from dotenv import load_dotenv
import os
import requests
import webuiapi

load_dotenv()

def save_image(data, file_path, save_file=False):
    # Strip out the header of the base64 string if present
    if ',' in data:
        header, data = data.split(',', 1)

    image = base64.b64decode(data)
    
    if save_file:
        with open(file_path, "wb") as fh:
            fh.write(image)

    return image

def generate_getimgai(mediaid, image_model, prompt):
    url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"

    payload = {
        "model": "stable-diffusion-v1-5",
        "prompt": prompt,
        "negative_prompt": "Disfigured, cartoon, blurry",
        "width": 512,
        "height": 512,
        "steps": 25,
        "guidance": 7.5,
        "seed": 0,
        "scheduler": "dpmsolver++",
        "output_format": "png"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": os.environ['GETIMG_API_KEY']
    }

    try:
        response = requests.post(url, json=payload, headers=headers)

        print(response.image)
        return response.image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def generate_sd_webui(mediaid, prompt, save_file=False):
    try:
        result = sdui_api.txt2img(prompt=prompt,
                            negative_prompt=args.negative_prompt,
                            save_images=False,
                            width=512,
                            height=512,
        #                    seed=1003,
        #                    styles=["anime"],
        #                    cfg_scale=7,
        #                      sampler_index='DDIM',
        #                      steps=30,
        #                      enable_hr=True,
        #                      hr_scale=2,
        #                      hr_upscaler=webuiapi.HiResUpscaler.Latent,
        #                      hr_second_pass_steps=20,
        #                      hr_resize_x=1536,
        #                      hr_resize_y=1024,
        #                      denoising_strength=0.4,

                )

        sdui_api.util_wait_for_ready()

        if result.image is not None:
            if save_file:
                result.image.save(f"images/{mediaid}.png")
                logger.info(f"Saved image to images/{mediaid}.png")
                print(f"Saved image to images/{mediaid}.png")
        else:
            logger.error(f"Error generating image: {result.error}")
            return None
        
        return result.image
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

def generate_openai(mediaid, image_model, prompt, username="lifeai", return_url=False, save_file=False):
    response = openai_client.images.generate(
        model=image_model,
        prompt=prompt,
        size=f"{args.width}x{args.height}",
        quality=args.quality,
        style=args.style,
        response_format="b64_json",
        user=username,
        n=1,
    )

    logger.debug(f"{response.data[0]}")

    image_url = response.data[0].url
    b64_json = response.data[0].b64_json

    revised_prompt = response.data[0].revised_prompt
    logger.info(f"OpenAI revised prompt: {revised_prompt}")

    image = save_image(b64_json, f"images/{mediaid}.png", save_file)
    if return_url:
        print(f"got url: {image_url}")
    
    return image

trlogging.set_verbosity_error()

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove image tags or Markdown image syntax
    text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<img.*?>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove any inline code blocks
    text = re.sub(r'`.*?`', '', text)
    
    # Remove any block code segments
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove special characters and digits (optional, be cautious)
    text = re.sub(r'[^a-zA-Z0-9\s.?,!\n]', '', text)

    # This seems to provoke some questionable images :/
    text = text.replace("black friday", "good friday").replace("Black Friday", "good friday").replace("black Friday", "good friday").replace("Black friday", "good friday")
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def main():
    last_image = None
    last_image_time = 0
    retry = False
    latency = 0
    max_latency = args.max_latency
    throttle = False
    header_message = None
    skipped_messages = 0
    while True:
        if throttle and args.max_latency > 0:
            start = time.time()
            combine_time = 0
            if max_latency > 0:
                combine_time = max(0, (latency / 1000) - max_latency)

            # read and combine the messages for 60 seconds into a single message
            priority = 0
            while max_latency > 0 and time.time() - start < combine_time:
                header_message = receiver.recv_json()
                header_message["stream"] = "image"
                header_message["throttle"] = "true"
                if 'priority' in header_message:
                    priority = header_message["priority"]
                    if priority == 100:
                        retry = True # keep header and continue with this message on loop
                        break

                sender.send_json(header_message, zmq.SNDMORE)
                sender.send(last_image)

            logger.info(f"TTI: Throttling for {combine_time} seconds.")

        # Receive a message
        if retry:
            logger.error(f"Retrying...")
            retry = False
        else:
            header_message = receiver.recv_json()

        # get variables from header
        mediaid = header_message["mediaid"]
        segment_number = header_message["segment_number"]
        header_message["throttle"] = "false"
        optimized_prompt = ""
        if "optimized_text" in header_message and header_message["optimized_text"] != "":
            optimized_prompt = header_message["optimized_text"]
        else:
            optimized_prompt = header_message["text"]
            logger.warning(f"TTI: No optimized text, using original text.")

        # genre
        genre = args.genre
        if "genre" in header_message and header_message["genre"] != "":
            genre = header_message["genre"]

        image = None
        speaker_pattern = r'(?:(?:\[/INST\])?<<([A-Za-z0-9_\)\(\-]+)>>|^(?:\[\w+\])?([A-Za-z0-9_\)\(\-)]+):)'
        speaker_line = False
        speaker_name = ""
        # Find speaker names in the text and derive gender from name, setup speaker map
        for line in optimized_prompt.split('\n'):
            speaker_match = re.search(speaker_pattern, line)
            if speaker_match:
                # Extracting speaker name from either of the capturing groups
                new_speaker = speaker_match.group(1) or speaker_match.group(2)
                new_speaker = new_speaker.strip()
                new_speaker = new_speaker.lower()
                speaker_line = True
                speaker_name = new_speaker
                break

        # Clean text
        optimized_prompt_clean = clean_text(optimized_prompt)

        # create prompt
        optimized_prompt_final = f"{speaker_name} {genre[:30]} {header_message['message'][:80]} {optimized_prompt_clean[:200]}"

        logger.debug(
            f"Text to Image recieved optimized prompt:\n{header_message}.")
        logger.info(
            f"Text to Image using text as prompt #{segment_number}:\n - {optimized_prompt_final[:80]}...")

        if (skipped_messages >= 2 or speaker_line or last_image == None) and (args.wait_time == 0 or last_image == None or time.time() - last_image_time >= args.wait_time):
            skipped_messages = 0
            if args.service == "openai":
                image = generate_openai(mediaid, args.oai_image_model, optimized_prompt_final, header_message["username"], args.save_images)
            elif args.service == "sdwebui":
                image = generate_sd_webui(mediaid, optimized_prompt_final, args.save_images)
            elif args.service == "getimgai":
                image = generate_getimgai(mediaid, args.sdwebui_image_model, optimized_prompt_final)
            else:
                if args.extend_prompt:
                    max_length = pipe.tokenizer.model_max_length

                    # 3. Forward
                    input_ids = pipe.tokenizer(optimized_prompt_final, return_tensors="pt").input_ids
                    input_ids = input_ids.to("mps")

                    negative_ids = pipe.tokenizer("", truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
                    negative_ids = negative_ids.to("mps")

                    concat_embeds = []
                    neg_embeds = []
                    for i in range(0, input_ids.shape[-1], max_length):
                        concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
                        neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

                    prompt_embeds = torch.cat(concat_embeds, dim=1)
                    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
                    
                    # 2. Forward embeddings and negative embeddings through text encoder
                    image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds).images[0]
                else:
                    image = pipe(optimized_prompt_final).images[0]

            if image != None:
                if args.service != "openai": # and args.service != "sdwebui":
                    # Convert PIL Image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')  # Save it as PNG or JPEG depending on your preference
                    image = img_byte_arr.getvalue()
                #elif args.service == "sdwebui":
                 #   image = image.copy()

                # check if image is more than 75k
                if args.service != "openai" and args.service != "sdwebui" and len(image) < 75000:
                    logger.error(f"Image is too small, retrying...")
                    retry = True
                    continue

                last_image = image
                last_image_time = time.time()
            else:
                logger.error(f"Error generating image, retrying...")
                retry = True
                continue
        else:
            header_message["throttle"] = "true"
            skipped_messages += 1

        header_message["stream"] = "image"

        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(last_image)

        logger.info(f"Text to Image sent image #{segment_number} {header_message['timestamp']} of {len(last_image)} bytes.")

        # measure latency and see if we need to throttle output
        if args.service != "openai":
            latency = round(time.time() * 1000) - header_message['timestamp']
            if latency > (max_latency * 1000) and max_latency > 0:
                logger.error(f"TTI: Message is too old {latency/1000}, throttling for the next{latency/1000} seconds.")
                throttle = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2000, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=6002, required=False, help="Port for sending image output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Port for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Port for sending image output")
    parser.add_argument("--nsfw", action="store_true", default=False, help="Disable NSFW filters, caution!!!")
    parser.add_argument("--metal", action="store_true", default=False, help="offload to metal mps GPU")
    parser.add_argument("--cuda", action="store_true", default=False, help="offload to metal cuda GPU")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--hg_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Huggingface Model ID to use, default unwayml/stable-diffusion-v1-5")
    parser.add_argument("--wait_time", type=int, default=0, help="Time in seconds to wait between image generations")
    parser.add_argument("--extend_prompt", action="store_true", help="Extend prompt past 77 token limit.")
    parser.add_argument("--max_latency", type=int, default=0, help="Max latency for messages before they are throttled / combined")
    parser.add_argument("--service", type=str, default="sdwebui", help="Service to use for image generation: huggingface, openai, sdwebui, getimgai")
    parser.add_argument("--save_images", action="store_true", help="Save images to disk")
    parser.add_argument("--oai_image_model", type=str, default="dall-e-2", help="OpenAI image model to use")
    parser.add_argument("--sdwebui_image_model", type=str, default="sd_xl_turbo", help="Local SD WebUI API Image model to use, default protogenV2")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--style", type=str, default="vivid", help="Image style for dalle-3, standard or vivid")
    parser.add_argument("--quality", type=str, default="standard", help="Image quality for dalle-3, standard or hd")
    parser.add_argument("--webui_url", type=str, default="127.0.0.1:7860", help="URL for webui, default 127.0.0.1:7860")
    parser.add_argument("--genre", type=str, default="", help="Genre for the model")
    parser.add_argument("--negative_prompt", type=str, default="Disfigured, cartoon, blurry, nsfw, naked, porn, violence, gore, racism, black face", help="Negative prompt for the model")

    args = parser.parse_args()

    LOGLEVEL = logging.INFO

    if args.loglevel == "info":
        LOGLEVEL = logging.INFO
    elif args.loglevel == "debug":
        LOGLEVEL = logging.DEBUG
    elif args.loglevel == "warning":
        LOGLEVEL = logging.WARNING
    else:
        LOGLEVEL = logging.INFO

    log_id = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f"logs/tti-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('TTI')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    model_id = args.hg_model

    ## Disable NSFW filters
    pipe = None
    if args.service == "huggingface":
        if args.nsfw:
            pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                            torch_dtype=torch.float16,
                                                            safety_checker = None,
                                                            requires_safety_checker = False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

        ## Offload to GPU Metal
        if args.metal:
            pipe = pipe.to("mps")
        elif args.cuda:
            pipe = pipe.to("cuda")
        else:
            pipe = pipe.to("mps")

    sdui_api = None
    if args.service == "sdwebui":
        # create API client with custom host, port
        host, port = args.webui_url.split(":")
        sdui_api = webuiapi.WebUIApi(
            host='127.0.0.1', 
            port=7860,
            use_https=False)

        if args.loglevel == "debug":
            sdui_api.refresh_checkpoints()
            models = sdui_api.util_get_model_names()
            print(f"Available models: {models}")
            current_model = sdui_api.util_get_current_model()
            print(f"Current model: {current_model} requested model: {args}")
            logger.info(f"Current model: {current_model} requested model: {args.sdwebui_image_model}")

        sdui_api.util_set_model(args.sdwebui_image_model)

    openai_client = None
    if args.service == "openai":
        openai_client = OpenAI()
        if args.wait_time == 0:
            args.wait_time = 60

    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    logger.info("connected to ZMQ in: %s:%d" % (args.input_host, args.input_port))
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    sender = context.socket(zmq.PUSH)
    logger.info("binded to ZMQ out: %s:%d" % (args.output_host, args.output_port))
    sender.connect(f"tcp://{args.output_host}:{args.output_port}")
    main()

