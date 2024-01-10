from pathlib import Path
import sys, os
import torch
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE, DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer, ChineseTokenizer

from torchvision.utils import make_grid, save_image
from PIL import Image

'''
Usage: python dalle_gen.py [text] [img_num] [save_path]
'''
if len(sys.argv) == 4:
    text = sys.argv[1]
    img_num = int(sys.argv[2])
    save_path = sys.argv[3]
else:
    print('Usage: python dalle_gen.py [text] [img_num] [save_path]')
    exit(1)


def exists(val):
    return val is not None

# constants
VAE_PATH = './dVae_model/vae_8192tk_3ly_128sz.pt'

taming = True
VQGAN_MODEL_PATH = './VQGAN/last.ckpt'
VQGAN_CONFIG_PATH = './VQGAN/model.yaml'
DALLE_PATH = './dalle_save/vqgan_dalle_dep8_ep70.pt'
# DALLE_PATH = './dalle.pt'
RESUME = exists(DALLE_PATH)

# reconstitute vae

dalle_path = Path(DALLE_PATH)

assert dalle_path.exists(), 'DALL-E model file does not exist'
loaded_obj = torch.load(str(dalle_path), map_location='cpu')

dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
opt_state = loaded_obj.get('opt_state')
scheduler_state = loaded_obj.get('scheduler_state')

if taming:
    vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
elif vae_params is not None:
    vae = DiscreteVAE(**vae_params)
else:
    vae = OpenAIDiscreteVAE()

IMAGE_SIZE = vae.image_size
resume_epoch = loaded_obj.get('epoch', 0)


# initialize DALL-E

dalle = DALLE(vae=vae, **dalle_params)
dalle = dalle.cuda()
dalle.load_state_dict(weights)




sample = text
tokenizer = ChineseTokenizer()
mysample = tokenizer.tokenize(sample,32,truncate_text=None).cuda()

for i in range(img_num):
    images = dalle.generate_images(mysample, filter_thres=0.9)  # topk sampling at 0.9
    save_image(images[0], os.path.join(save_path, f'{i}.jpg'), normalize=True)


