# init pycord
print("Starting up pycord")
import discord
from discord.ext import commands
intents = discord.Intents.all()
intents.message_content = True
bot = commands.Bot(command_prefix="-", intents=intents)

# auth etc
print("Authenticating...")
import os, time, random
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
random.seed(time.time())
from PIL import Image, ImageOps
import requests
from io import BytesIO
from colorthief import ColorThief
import numpy as np
import torch

# gpt3
import openai
openai.api_key = os.getenv('OPENAI_TOKEN')
def call_gpt3(prompt):
    response = openai.Completion.create(model='text-davinci-002', prompt=prompt, temperature=0.75, max_tokens=100)
    print(response.choices[0].text.strip())
    return response.choices[0].text
    
def kprompt_call(kwords):
    print('asking gpt3 to enhance...')
    # enhance is designed to take a couple of words and write a sentence or two full of visually descriptive text
    # it also works well to juice up a regular prompt instead of just using keywords
    # it likes to be grammatically correct more than it likes to spam keywords like stable diffusion prompters lol
    kwords_prompt = """Take a couple of keywords and write a sentence or two full of visually descriptive text for each one.\n
    Here are some examples:\n
    keywords: hat guitar ocean\n
    prompt: a black and white photograph of a man wearing a hat playing a guitar in the ocean, hd 4k hyper detailed\n\n
    keywords: sunrise meadow clouds\n
    prompt: a beautiful sunrise in a meadow, surrounded by clouds, beautiful painting, by rebecca guay, yoshitaka amano, trending on artstation hq\n\n
    keywords: lawyer\n
    prompt: a beefy intimidating copyright lawyer with glowing red eyes, dramatic portrait, digital painting portrait, ArtStation\n\n
    keywords: marshmallow eiffel\n
    prompt: giant pink marshmallow man stomping towards the eiffel tower, film scene, design by Pixar and WETA Digital, 8k photography\n\n
    keywords: girl plants ghibli\n
    prompt: a girl watering her plants, studio ghibli, art by hayao miyazaki, artstation hq, wlop, by greg rutkowski, ilya kuvshinov \n\n
    keywords: psychedelic\n
    prompt: ego death, visionary artwork by Alex Grey, hyperdetailed digital render, fractals, dramatic, 3d render\n\n
    Make sure to include style annotations in your prompt, such as '8k photograph', 'digital art', 'oil painting', 'realistic, detailed', 'portrait', and 'by '\n
    Now it's your turn! \n
    Make a detailed prompt for the following keywords.\n
    keywords:	"""+kwords+"""\n
    prompt: """
    response = call_gpt3(kwords_prompt)
    # response = response.split("keywords:")[0] if 'keywords:' in response else response
    if response.strip() == '':
        return kprompt_call(kwords)
    print(response)
    return response

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

original_conv2d_init = torch.nn.Conv2d.__init__

def patch_conv(**patch):
    global original_conv2d_init
    cls = torch.nn.Conv2d
    init = cls.__init__
    original_conv2d_init = init
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    cls.__init__ = __init__

def reset_conv():
    if original_conv2d_init is None:
        print('Resetting convolution failed, no original_conv2d_init')
    cls = torch.nn.Conv2d
    cls.__init__ = original_conv2d_init

# load stable diffusion
import torch
from torch import autocast
import gc

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler

model_name = "./stable-diffusion-v1-4"

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
offload_device = "cpu"

# soon
# import open_clip as clip
# clip_model = clip.create_model("ViT-B-32", pretrained="laion2b_s34b_b79k")

vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
try:
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
except:
    print("Text encoder could not be loaded from the repo specified for some reason, falling back to the vit-l repo")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

vae = vae.to(offload_device).half()
text_encoder = text_encoder.to(offload_device).half()
unet = unet.to(torch_device).half()

def requires_grad(model, val=False):
    for param in model.parameters():
        param.requires_grad = val

requires_grad(vae)
requires_grad(text_encoder)
requires_grad(unet)

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):

    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)

    text_encoder.resize_token_embeddings(len(tokenizer))
    
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    return text_encoder, tokenizer


print('loading concepts....')
for concept in os.listdir('concepts'):
    token_name = f'<{concept.split(".")[0]}>'
    print(f'loading {token_name}')
    text_encoder, tokenizer = load_learned_embed_in_clip(f'concepts/{concept}', text_encoder, tokenizer, token_name)

print('done')

loaded_model = 'text2img'
last_used = time.time()

print("Loading stable diffusion pipeline")
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-4",
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    revision="fp16",
    torch_dtype=torch.float16)
pipe = pipe.to("cuda")

torch.manual_seed(0)

def autoresize(img, max_p):
    img_width, img_height = img.size
    total_pixels = img_width*img_height
    if total_pixels > max_p:
        import math
        ratio = 632 / math.sqrt(total_pixels)
        new_size = [64 * int(ratio * s) % 64 for s in img.size]
        img = img.resize( new_size )
        return img
    return img

def alpha_to_mask(img):
    alpha = img.split()[-1]
    bg = Image.new("RGBA", img.size, (0,0,0,255))
    bg.paste(alpha, mask=alpha)
    bg = ImageOps.invert(bg.convert('RGB'))
    img = img.convert('RGB')
    return (img, bg)

# -dream command
@bot.command()
async def dream(ctx, *prompt):
    global loaded_model, pipe, last_used

    # parse prompt and keywords
    arg_words = [t for t in prompt if '=' in t]
    kwargs = dict(t.split('=') for t in arg_words) if arg_words is not [] else {}
    prompt = ' '.join([t for t in prompt if '=' not in t])

    # get attachment images
    if len(ctx.message.attachments) > 0:
        print(f'saw attachments: {ctx.message.attachments}')
        response = requests.get(ctx.message.attachments[0].url)
        input_img = Image.open(BytesIO(response.content)).convert('RGBA')
        input_img = autoresize(input_img, 380000)

        if len(ctx.message.attachments) > 1:
            response = requests.get(ctx.message.attachments[1].url)
            mask = Image.open(BytesIO(response.content)).convert('RGB')
            input_img = input_img.convert('RGB')
        else:
            input_img, mask = alpha_to_mask(input_img)

        kwargs['init_img'] = input_img
        kwargs['mask'] = mask

    # start generating
    n_images = int(kwargs['n']) if 'n' in kwargs else 1
    for i in range(n_images):

        seed = int(kwargs['seed']) if 'seed' in kwargs else random.randrange(0, 2**32)
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f'\"{prompt}\" by {ctx.author.name}')
        print(f'kwargs: {kwargs}')
        await ctx.send(f"starting dream for `{prompt}` with seed {seed} ({i+1}/{n_images})")
        
        start_time = time.time()
        try:
            print(f'')
            with autocast("cuda"):
                if 'init_img' in kwargs.keys():
                    if 'inpaint' in kwargs.keys():
                        if loaded_model != 'inpaint':    
                            await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model inpaint` and try again.')
                            continue
                        image = pipe(prompt,
                            init_image=kwargs['init_img'],
                            mask_image=kwargs['mask'],
                            generator=generator,
                            strength=float(kwargs['strength']) if 'strength' in kwargs else 0.75,
                            num_inference_steps=int(kwargs['steps']) if 'steps' in kwargs else 50,
                            guidance_scale=float(kwargs['scale']) if 'scale' in kwargs else 7.5).images[0]
                    else:
                        if loaded_model != 'img2img':
                            await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model img2img` and try again.')
                            continue
                        image = pipe(prompt,
                            init_image=kwargs['init_img'],
                            generator=generator,
                            strength=float(kwargs['strength']) if 'strength' in kwargs else 0.75,
                            num_inference_steps=int(kwargs['steps']) if 'steps' in kwargs else 50,
                            guidance_scale=float(kwargs['scale']) if 'scale' in kwargs else 7.5).images[0]
                else:
                    if loaded_model != 'seamless' and loaded_model != 'text2img':
                        await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model text2img` and try again.')
                        continue
                    image = pipe(prompt, 
                        generator=generator,
                        height=int(kwargs['height']) if 'height' in kwargs else 512,
                        width=int(kwargs['width']) if 'width' in kwargs else 512,
                        num_inference_steps=int(kwargs['steps']) if 'steps' in kwargs else 50,
                        guidance_scale=float(kwargs['scale']) if 'scale' in kwargs else 7.5).images[0]
                    
        except Exception as e:
            await ctx.send(f'error generating: {e}')
            return
        
        gen_count = len(os.listdir('outputs'))
        filename = f'outputs/{gen_count}.png'
        image.save(filename)
        elapsed_time = int(time.time() - start_time)
        await ctx.send(f"\"{prompt}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed} ({i+1}/{n_images})", file=discord.File(filename))
        
        last_used = time.time()
        # # reaction = (<Reaction emoji='🖼️' me=False count=1>, <Member id=891221733326090250 name='bleepybloops' discriminator='6448' bot=False nick='bleep bloop' guild=<Guild id=804209375098568724 name='Creativity Farm' shard_id=0 chunked=False member_count=29>>)
        # def check(reaction, user):
        #     return user == ctx.message.author
        # reaction = await bot.wait_for("reaction_add", check=check)
        # ai_art_channel_id = 907829317575245884
        # if '🖼' in reaction[0].emoji:
        #     print('reacted with 🖼')
        #     await bot.get_channel(ai_art_channel_id).send(f"\"{prompt}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed} ({i+1}/{n_images})", file=discord.File(filename))

@bot.command()
async def palette(ctx, *prompt):
    global loaded_model, pipe, last_used
    
    arg_words = [t for t in prompt if '=' in t]
    kwargs = dict(t.split('=') for t in arg_words) if arg_words is not [] else {}
    prompt = ' '.join([t for t in prompt if '=' not in t])
    
    import math
    n_colors = math.floor(float(kwargs['colors'])) if 'colors' in kwargs else 5
    n_colors = (n_colors if n_colors > 3 else 4) if n_colors < 8 else 7

    n_images = int(kwargs['n']) if 'n' in kwargs else 1
    for i in range(n_images):
        seed = int(kwargs['seed']) if 'seed' in kwargs else random.randrange(0, 2**32)
        generator = torch.Generator("cuda").manual_seed(seed)
        print(seed)
        await ctx.send(f"making color palette for `{prompt}` with seed {seed}")

        if loaded_model != 'seamless' and loaded_model != 'text2img':        
            await ctx.send(f'currently loaded model: {loaded_model}. please run `-load_model text2img` and try again.')
            return
        print(loaded_model)
        print('starting...')

        start_time = time.time()
        with autocast("cuda"):
            image = pipe(prompt, 
                generator=generator,
                height=int(kwargs['height']) if 'height' in kwargs else 384,
                width=int(kwargs['width']) if 'width' in kwargs else 384,
                num_inference_steps=int(kwargs['steps']) if 'steps' in kwargs else 30,
                guidance_scale=float(kwargs['scale']) if 'scale' in kwargs else 7.5).images[0]
        print(image)

        gen_count = len(os.listdir('outputs'))
        filename = f'outputs/{gen_count}.png'
        image.save(filename)

        color_thief = ColorThief(filename)
        palette = color_thief.get_palette(color_count=n_colors, quality=10)

        hex_colors = ' '.join(f'#{rgb_to_hex(v).upper()}' for v in palette)
        palette = np.uint8(palette).reshape(1, n_colors, 3)
        pal_img = Image.fromarray(palette).resize((64*n_colors, 64), Image.Resampling.NEAREST)

        gen_count = len(os.listdir('outputs/palettes'))
        pal_img_fname = f'outputs/palettes/{gen_count}.png'
        pal_img.save(pal_img_fname)

        thumb = image.resize((64, 64))
        gen_count = len(os.listdir('outputs/thumbs'))
        thumb_fname = f'outputs/thumbs/{gen_count}.png'
        thumb.save(thumb_fname)

        elapsed_time = int(time.time() - start_time)
        await ctx.send(f"color palette for \"{prompt}\" by {ctx.author.mention} in {elapsed_time}s with seed {seed} ({i+1}/{n_images})\n{hex_colors}", files=[discord.File(pal_img_fname), discord.File(thumb_fname)])

@bot.command()
async def load_model(ctx, model):
    global loaded_model, pipe
    global unet, scheduler, vae, text_encoder, tokenizer
    if model == 'text2img':
        if loaded_model == 'text2img':
            return
        await ctx.send('Loading stable diffusion text2img pipeline...')
        pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        reset_conv()
        from diffusers import StableDiffusionPipeline
        patch_conv(padding_mode='zeros')
        pipe = StableDiffusionPipeline.from_pretrained(
            "./stable-diffusion-v1-4",
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            revision="fp16",
            torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        torch.manual_seed(0)
        loaded_model = 'text2img'
        await ctx.send('text2img model loaded. happy generating!')

    elif model == 'img2img':
        if loaded_model == 'img2img':
            return
        await ctx.send('Loading stable diffusion img2img pipeline...')
        pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        reset_conv()
        from diffusers import StableDiffusionImg2ImgPipeline
        patch_conv(padding_mode='zeros')
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "./stable-diffusion-v1-4",
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            revision="fp16",
            torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        torch.manual_seed(0)
        loaded_model = 'img2img'
        await ctx.send('img2img model loaded. happy generating!')

    elif model == 'inpaint':
        if loaded_model == 'inpaint':
            return
        await ctx.send('Loading stable diffusion inpainting pipeline...')
        pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        reset_conv()
        from diffusers import StableDiffusionInpaintPipeline
        patch_conv(padding_mode='zeros')
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "./stable-diffusion-v1-4",
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            revision="fp16",
            torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        torch.manual_seed(0)
        loaded_model = 'inpaint'
        await ctx.send('inpainting model loaded. happy generating!')

    elif model == 'seamless':
        if loaded_model == 'seamless':
            return
        await ctx.send('Loading stable diffusion seamless text2img pipeline...')
        pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        reset_conv()
        from diffusers import StableDiffusionPipeline
        patch_conv(padding_mode='circular')
        pipe = StableDiffusionPipeline.from_pretrained(
            "./stable-diffusion-v1-4",
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            revision="fp16",
            torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        torch.manual_seed(0)
        loaded_model = 'seamless'
        await ctx.send('seamless model loaded. happy generating!')

    else:
        await ctx.send(f'model {model} not found: try using \'text2img\', \'img2img\', \'inpaint\', or \'seamless\'')

@bot.command()
async def clear_cuda_mem(ctx):
    global pipe, torch
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    await ctx.send('cuda cleared')

@bot.command()
async def last_used(ctx):
    global last_used
    since = int(time.time() - last_used)
    await ctx.send(f'last used {since}s ago')

# -gpt command
@bot.command()
async def gpt(ctx, *prompt):
    await ctx.send(call_gpt3(' '.join(prompt)))

# -enhance command
@bot.command()
async def enhance(ctx, *prompt):
    await ctx.send(kprompt_call(' '.join(prompt)))

@bot.command()
async def upscale(ctx, *prompt):
    print('upscaling')
    response = requests.get(ctx.message.attachments[0].url)
    input_img = Image.open(BytesIO(response.content)).convert('RGB')
    import torch
    from esr.realesrgan import RealESRGAN
    device1 = torch.device('cuda')
    esr_model = RealESRGAN(device1, scale=4)
    esr_model.load_weights('esr\weights\RealESRGAN_x4.pth')
    esr_image = esr_model.predict(input_img)
    upscaled_count = len(os.listdir('outputs/upscaled'))
    new_path = f'outputs/upscaled/{upscaled_count}.png'
    esr_image.save(new_path)
    print('done')
    await ctx.send(f"upscaled by {ctx.author.mention}", file=discord.File(new_path))

@bot.command()
async def tile(ctx, *prompt):
    arg_words = [t for t in prompt if '=' in t]
    kwargs = dict(t.split('=') for t in arg_words) if arg_words is not [] else {}
    prompt = ' '.join([t for t in prompt if '=' not in t])

    target_x = int(kwargs['x']) if 'x' in kwargs else 4
    target_y = int(kwargs['y']) if 'y' in kwargs else 4

    response = requests.get(ctx.message.attachments[0].url)
    input_img = Image.open(BytesIO(response.content)).convert('RGB')

    w, h = int(input_img.size[0]/2), int(input_img.size[1]/2)
    input_img = input_img.resize((w, h))

    img_collage = Image.new('RGB', (w*target_x, h*target_y))
    for i in range(target_x):
        for j in range(target_y):
            img_collage.paste(input_img, (i*w, j*h))

    tile_count = len(os.listdir('outputs/tiled'))
    tiled_fname = f'outputs/tiled/{tile_count}.png'
    img_collage.save(tiled_fname)

    await ctx.send(f'tiled by {ctx.author.mention}', file=discord.File(tiled_fname))

@bot.command()
async def interrogate(ctx):
    await ctx.send('starting clip-interrogator...')
    print('Interrogating...')
    from clip_interrogator.clip_interrogator import interrogate
    response = requests.get(ctx.message.attachments[0].url)
    input_img = Image.open(BytesIO(response.content)).convert('RGB')
    response = interrogate(input_img, models=['ViT-L/14'])
    print('done')
    await ctx.send(f'clip-interrogator thinks your picture looks like `{response}`')

@bot.command()
async def get_concept(ctx, conceptname):
    global tokenizer, text_encoder

    downloadurl = f'https://huggingface.co/sd-concepts-library/{conceptname}/resolve/main/learned_embeds.bin'
    print(downloadurl)

    # download downloadurl into "concepts/{conceptname}.bin"
    response = requests.get(downloadurl)

    with open(f'concepts/{conceptname}.bin', 'bw') as f:
        f.write(response.content)
        f.close()

    token_name = f'<{conceptname.split(".")[0]}>'
    print(f'loading {token_name}')
    text_encoder, tokenizer = load_learned_embed_in_clip(f'concepts/{conceptname}.bin', text_encoder, tokenizer, token_name)
    
    await ctx.send(f'finished downloading `<{conceptname}>` from `huggingface.co/sd-concepts-library`')
    

# go!
print("Connected")
bot.run(TOKEN)
