import torch
from torchvision import transforms as T
from pathlib import Path
import os
from tqdm import tqdm
from dalle_pytorch import OpenAIDiscreteVAE, DALLE
from dalle_pytorch.tokenizer import SimpleTokenizer
import pandas as pd

# Change your input size here
input_image_size = 256

# Change your test image root path here
test_img_path = "./Flower_Dataset_Combine/ImagesCombine/"

# Change your test annot csv path here
test_annot_path = "./Flower_Dataset_Combine/New_captions.csv"

# Change your device ("cpu" or "cuda")
device = "cuda"

# Change your dalle model path here
dalle_load_path = "./dalle.pth"

# Change the test result image save path (should be a directory or folder)
test_img_save_path = "./result"

if not os.path.exists(test_img_save_path):
    os.makedirs(test_img_save_path)

transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize(input_image_size),
    T.CenterCrop(input_image_size),
    T.ToTensor()
])

test_csv= pd.read_csv(test_annot_path)

test_csv = test_csv.drop_duplicates()
test_csv = test_csv.dropna()

tokenizer = SimpleTokenizer()

vae = OpenAIDiscreteVAE()

dalle = DALLE(
    dim = 1024,
    vae = vae,                                 # automatically infer (1) image sequence length and (2) number of image tokens
    num_text_tokens = tokenizer.vocab_size,    # vocab size for text
    text_seq_len = 256,                        # text sequence length
    depth = 1,                                 # should aim to be 64
    heads = 16,                                # attention heads
    dim_head = 64,                             # attention head dimension
    attn_dropout = 0.1,                        # attention dropout
    ff_dropout = 0.1                           # feedforward dropout
).to(device)

dalle.load_state_dict(torch.load(dalle_load_path))

for data in tqdm(test_csv.iterrows()):
    target = [data[1]['caption']]

    text = tokenizer.tokenize(target).to(device)

    test_img_tensors = dalle.generate_images(text)

    for test_idx, test_img_tensor in enumerate(test_img_tensors):
        test_img = T.ToPILImage()(test_img_tensor)
        test_save_path = test_img_save_path + "/" + str(target[test_idx]) + ".jpg"
        test_img.save(Path(test_save_path))