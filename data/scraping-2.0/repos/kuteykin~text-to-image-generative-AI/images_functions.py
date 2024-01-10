###########################################################
#   (C) Copyright 2023 by Konstantin Kuteykin-Teplyakov   #
#                  All Rights Reserved                    #
#           ANY COMMERCIAL USE IS NOT ALLOWED             #
#        Violation of IP rights will be prosecuted        #
#                                                         #
# DISCLAIMER: This code was disclosed to general public   #
# solely for informational purposes, and any commercial   #
# use as part of software or paid service is              #
# NOT PERMITTED without written consent of IP rights      #
# owner.                                                  #
###########################################################


def generate_images(
    image_prompt,
    MODEL_PATH,
    negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
    custom_model_path="",
    USER_DIR="",
    num_inference_steps=50,
    num_images_per_prompt=1,
    save_images=False,
):
    """
    Generates images using inference on Stable Diffusion models with Hugging Face Diffusers library.
    Receives prompts, paths and inference parameters, returns images encoded in Base64 format.
    Optionally can save generated images locally in PNG files.
    """
    # Load necessary libraries
    import base64
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    from flask import jsonify
    import io
    import logging
    from pathlib import Path
    import torch

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Convert paths to PosixPath
    MODEL_PATH = Path(MODEL_PATH)
    USER_DIR = Path(USER_DIR)
    custom_model_path = Path(custom_model_path)

    # Load the Stable Diffusion Model
    if MODEL_PATH.is_dir():
        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline.to("cuda")
        pipeline.enable_xformers_memory_efficient_attention()
        logging.debug(f"Successfully loaded base model {MODEL_PATH}")
    else:
        logging.debug(
            "*** ERROR: Base SDiff model not found. *** \nPlease run *download_sdiff_model.py* to DOWNLOAD the base model and save it locally.\n *** EXIT! *** \n"
        )
        return jsonify({"error": "Base model not found  - EXIT!"}), 400

    if custom_model_path.is_file():
        logging.debug(f"Custom model used: {custom_model_path}")
        model_name = custom_model_path.name
        pipeline.unet.load_attn_procs(custom_model_path)
    else:
        model_name = MODEL_PATH.name

    # Generate images using the pipeline
    print("\n*** COMMERCIAL USE IS FORBIDDEN ***\n")
    # generator = torch.Generator(device="cuda").manual_seed(42)      # Added for reproducibility for testing purpose
    created_images = pipeline(
        image_prompt,
        #              generator=generator,                               # Added for reproducibility for testing purpose
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        negative_prompt=negative_prompt,
    ).images

    # Save images to local drive
    if save_images:
        import slugify

        prompt_as_path = slugify.slugify(image_prompt)
        output_path = str(USER_DIR / "output" / model_name / prompt_as_path)[:255]
        Path(output_path).mkdir(parents=True, exist_ok=True)
        for idx, im in enumerate(created_images):
            im.save(f"{output_path}/{idx}.png")
        logging.debug(
            f"Images saved on local drive in [{USER_DIR}/output/{model_name}/{prompt_as_path}] folder"
        )

    # Convert the PIL Image objects to bytes
    images = []
    for i in range(num_images_per_prompt):
        img_byte_arr = io.BytesIO()
        created_images[i].save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
        images.append(img_base64)

    return images


def ask_chatgpt(input, task, dict):
    """
    Receives text input, task and a dictionary with task-specific prompts,
    sends request to OpenAI API and returns a ChatGPT answer as plain text
    """
    # Load necessary libraries
    import openai
    import os

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Generate ChatGPT prompt
    chatgpt_prompt = (
        f"{dict[task]['prompt_prefix']} {input} {dict[task]['prompt_suffix']}"
    )
    print("\n*** COMMERCIAL USE IS FORBIDDEN ***\n")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": chatgpt_prompt}],
        temperature=dict[task]["temperature"],
    )
    return response["choices"][0]["message"]["content"]
