import clip
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import kornia.augmentation as K
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)

print("Loading clip")
clip_model, clip_preprocess = clip.load("ViT-B/32", jit=False)
print("Loading clip done!")

# works with np, but the clip one assumes PIL
clip_norm = Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)

clip_res = 224


@torch.no_grad()
def get_image_embedding(img):
    x = clip_preprocess(img)
    x = clip_model.encode_image(x.cuda()[None])
    x /= x.norm(dim=-1, keepdim=True)
    return x


@torch.no_grad()
def get_text_embedding(classnames):
    zeroshot_weights = []
    for classname in classnames:
        texts = [
            template.format(classname) for template in imagenet_templates
        ]  # format with class
        texts = clip.tokenize(texts).cuda()  # tokenize
        class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
    return zeroshot_weights


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x

    return TF.to_tensor(x)


# slightly modified from OpenAI's code, so that it works with np tensors
# see https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/clip.py#L58
clip_preprocess = Compose(
    [
        to_tensor,
        Resize(clip_res, interpolation=Image.BICUBIC),
        CenterCrop(clip_res),
        clip_norm,
    ]
)

def clip_infer(x):
    return clip_model.encode_image(x)


def make_aug(x: torch.Tensor):
    if x.ndim < 4:
        x = x[None]

    x = x.repeat(8, 1, 1, 1)
    x = K.functional.random_affine(x, 30, (0.2, 0.2), (0.9, 1.5), [0.1, 0.4])
    x = K.functional.color_jitter(x, 0.2, 0.3, 0.2, 0.3)
    return x


@torch.no_grad()
def get_clip_code(img, use_aug=False):
    x = TF.to_tensor(img).cuda()
    if use_aug:
        x = make_aug(x)
    else:
        x = x[None]
    x = clip_preprocess(x)
    x = clip_infer(x)

    if use_aug:
        x = x.mean(axis=0, keepdim=True)

    # normalize since we do dot products lookups
    x /= x.norm()

    return x


