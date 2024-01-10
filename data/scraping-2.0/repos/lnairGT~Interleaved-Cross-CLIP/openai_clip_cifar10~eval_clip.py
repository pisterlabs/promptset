import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as CFG
from openai_clip_model import CLIP


def find_matches(image_embeddings, text_embeddings):
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = torch.softmax(100 * image_embeddings @ text_embeddings.T, dim=-1)
    pred = dot_similarity.argmax(dim=1, keepdim=True)
    return pred

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    val_bsz = train_bsz = 10

    model_path = "openai_CIFAR10.pt"

    model = CLIP(
        embed_dim=CFG.projection_dim,
        image_resolution=CFG.size,
        vision_layers=6,
        vision_width=512,
        vision_patch_size=4,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=8,
        transformer_layers=6
    )
    model = model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    # Prepare data and training entities
    from dataloaders import get_original_cifar10_dataloaders
    root = "/data/datasets"
    _, test_loader = get_original_cifar10_dataloaders(
        root, train_bsz=train_bsz, val_bsz=val_bsz
    )
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    from simple_tokenizer import load_and_transform_text
    tokens = load_and_transform_text(class_names, CFG.device)
    text_embeddings = model.encode_text(tokens)

    correct = 0
    with torch.no_grad():
        for batch, targets in tqdm(test_loader):
            targets = targets.to(device)
            batch = batch.to(device)
            image_embeddings = model.encode_image(batch)
            pred = find_matches(image_embeddings, text_embeddings)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"accuracy: {accuracy:.01f}" + "\n")

if __name__ == "__main__":
    main()
