import openai
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torchvision import utils as vutils
from torchvision.transforms.functional import adjust_contrast

from PIL import Image
import argparse

from utils.template import imagenet_templates
import utils.CLIP as CLIP
from utils.MODELS import UNet

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_path', type=str, default="./data/face2.jpeg",
                        help='Image resolution')
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='Image resolution')
    parser.add_argument('--text', type=str, default="Van_Gogh_Horus",
                        help='Image resolution')
    parser.add_argument("--output_path", type=str, default="./outputs/",
                        help="Result storage")
    parser.add_argument("--source", type=str, default="a Photo",
                        help="source in template")

    parser.add_argument('--lambda_tv', type=float, default=2e-3,
                        help='total variation loss parameter')
    parser.add_argument('--lambda_patch', type=float, default=9000,
                        help='PatchCLIP loss parameter')
    parser.add_argument('--lambda_dir', type=float, default=500,
                        help='directional loss parameter')
    parser.add_argument('--lambda_c', type=float, default=150,
                        help='content loss parameter')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='cropped image size')
    parser.add_argument('--num_crops', type=int, default=64,
                        help='number of patches')
    parser.add_argument('--img_width', type=int, default=512,
                        help='size of images')
    parser.add_argument('--img_height', type=int, default=512,
                        help='size of images')
    parser.add_argument('--max_step', type=int, default=120,
                        help='Number of domains')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Number of domains')
    parser.add_argument('--thresh', type=float, default=0.7,
                        help='Number of domains')

    parser.add_argument("--use_ChatGPT", type=bool, default=False,
                        help="Declare whether to use ChatGPT to expand text description")
    parser.add_argument("--max_expand_text", type=int, default=60,
                        help="Declare the maximum of text description that expanded by ChatGPT")

    args = parser.parse_args()
    return args


def load_image(img_path, img_height=None, img_width=None):
    image = Image.open(img_path)
    if img_width is not None:
        # change image size to (3, img_size, img_size)
        image = image.resize((img_width, img_height))  

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                  }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def img_normalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize(image, device):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


def expand_text(args):
    if args.use_ChatGPT:
        with open("./utils/config.txt", "r") as file:
            line = file.readlines()
        openai.api_key = line[0].split(sep=" = ")[-1]

        with_ChatGPT = "ChatGPT_"

        response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Describe " + args.text,
        temperature=0.3,
        max_tokens=args.max_expand_text,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )

        print(response["choices"][0]["text"])
        new_text = response["choices"][0]["text"]

    else:
        with_ChatGPT = ""
        new_text = args.text

    return new_text, with_ChatGPT


def train():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)
    style_net = UNet()
    style_net.to(device)
    clip_model, _ = CLIP.load('ViT-B/32', device, jit=False)

    # Freeze the network weights and do not update  while training
    for parameter in VGG.parameters():
        parameter.requires_grad_(False)

    # Initialize basic
    args = parse_args()
    assert (args.img_width % 8) == 0, "width must be multiple of 8"
    assert (args.img_height % 8) == 0, "height must be multiple of 8"
    content_image = load_image(args.content_path, img_height=args.img_height, img_width=args.img_width).to(device)
    content_features = get_features(img_normalize(content_image, device), VGG)
    target = content_image.clone().requires_grad_(True).to(device)
    output_image = content_image
    total_loss_epoch = []

    # Initialize tricks
    optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224)
    ])

    text, with_ChatGPT = expand_text(args)

    with torch.no_grad():
        template_text = compose_text_with_templates(text, imagenet_templates)
        tokens = CLIP.tokenize(template_text).to(device)
        text_features = clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        template_source = compose_text_with_templates(args.source, imagenet_templates)
        tokens_source = CLIP.tokenize(template_source).to(device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
        source_features = clip_model.encode_image(clip_normalize(content_image, device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

    for epoch in range(0, args.max_step + 1):
        content_loss = 0
        target = style_net(content_image, use_sigmoid=True).to(device)
        target.requires_grad_(True)
        target_features = get_features(img_normalize(target, device), VGG)

        loss_patch = 0
        img_proc = []
        for n in range(args.num_crops):
            target_crop = cropper(target)
            target_crop = augment(target_crop)
            img_proc.append(target_crop)

        img_proc = torch.cat(img_proc, dim=0)
        img_aug = img_proc

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # Start computing loss
        content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        image_features = clip_model.encode_image(clip_normalize(img_aug, device))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

        img_direction = (image_features - source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
        loss_temp[loss_temp < args.thresh] = 0
        loss_patch += loss_temp.mean()

        glob_features = clip_model.encode_image(clip_normalize(target, device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

        glob_direction = (glob_features - source_features)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

        loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

        reg_tv = args.lambda_tv * get_image_prior_losses(target)

        total_loss = args.lambda_patch * loss_patch + args.lambda_c * content_loss + reg_tv + args.lambda_dir * loss_glob
        total_loss_epoch.append(total_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # print loss in every 10 epoches
        if epoch % 10 == 0:
            print(f"epoch:{epoch}\tTotal loss: {total_loss.item()}\nContent loss: {loss_patch.item()}, Patch loss: {loss_patch.item()}\nDir loss:{loss_glob.item()}, TV loss: {reg_tv.item()}")

        # update target output in every 50 epoches
        if epoch % 50 == 0:
            out_path = args.output_path + args.text + '_' + args.content_path.split("/")[-1].split(".")[0] + '_' + with_ChatGPT + args.exp_name + '.jpg'
            output_image = torch.clamp(target.clone(), 0, 1)
            output_image = adjust_contrast(output_image, 1.5)
            vutils.save_image(output_image, out_path, nrow=1, normalize=True)


if __name__ == "__main__":
    train()
    