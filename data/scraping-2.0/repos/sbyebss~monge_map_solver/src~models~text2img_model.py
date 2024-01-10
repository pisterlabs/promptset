import os

import cv2
import torch
from dalle2_pytorch import OpenAIClipAdapter
from dalle2_pytorch.dalle2_pytorch import l2norm
from dalle2_pytorch.optimizer import get_optimizer
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid, save_image

from src.callbacks.txt2img_callbacks import generate_grid_samples
from src.datamodules.utils import split_test_full_data
from src.logger.jam_wandb import prefix_metrics_keys
from src.models.base_model import BaseModule
from src.models.loss_zoo import gradientOptimality
from src.viz.points import compare_highd_kde_scatter

# pylint: disable=abstract-method,too-many-ancestors,arguments-renamed,line-too-long,arguments-differ,unused-argument,too-many-locals

sampling_labels = [
    "Text emb.",
    "Our emb.",
    "laion emb.",
    "Real emb.",
    "Real Images",
]

# Function to add sampling_labels to an image
def add_labels_to_image(image_path):
    # Load the image
    with Image.open(image_path) as image:
        draw = ImageDraw.Draw(image)
        # Choose a font size
        font_size = 30
        # Load a font
        font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
        font = ImageFont.truetype(font_path, size=font_size)

        # Get image dimensions
        img_width, img_height = image.size
        # Calculate the height required for the text
        text_height = max(
            [draw.textsize(label, font=font)[1] for label in sampling_labels]
        )
        # Create a new image with extra space for the text labels
        new_img = Image.new(
            "RGB", (img_width, img_height + text_height + 20), (255, 255, 255)
        )  # White background for the new space
        # Paste the original image onto the new image
        new_img.paste(image, (0, text_height + 20))

        # Initialize ImageDraw to draw on the new image
        draw = ImageDraw.Draw(new_img)

        # Define the starting Y position for the text
        text_y = 10  # Small padding from the top of the new image
        # Calculate the width of a single column assuming sampling_labels are evenly spaced
        column_width = img_width / len(sampling_labels)

        # Iterate over the sampling_labels and their respective column positions
        for idx, label in enumerate(sampling_labels):
            # Calculate the position for each label (centered above each column)
            text_width, text_height = draw.textsize(label, font=font)
            text_x = idx * column_width + (column_width - text_width) / 2
            # Draw the text on the new image
            draw.text((text_x, text_y), label, font=font, fill=(0, 0, 0))  # Black text

        # Save the new image
        new_img.save(image_path)


class Text2ImgModule(BaseModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.clip = OpenAIClipAdapter(cfg.clip_model)
        self.image_embed_scale = cfg.image_embed_dim**0.5

    def get_real_data(self, batch):
        src_data, trg_data = batch
        # src_data: (image_embedding, tokenized_caption)
        # trg_data: (image_embedding, tokenized_caption)
        src_tokens, trg_img_emb = src_data[1], trg_data[0]
        text_embed, text_encodings = self.clip.embed_text(src_tokens)
        unnorm_text_embed = self.clip.clip.encode_text(src_tokens)
        src_text_cond = {
            "text_embed": text_embed,
            "text_encodings": text_encodings,
            "unnorm_text_embed": unnorm_text_embed,
        }
        trg_img_emb *= self.image_embed_scale
        return src_text_cond, trg_img_emb

    def loss_f(self, src_text_cond, trg_img_emb, mask=None):
        with torch.no_grad():
            tx_tensor = self.map_t(**src_text_cond)
        # assert torch.isclose(tx_tensor.norm(dim=-1).mean(), trg_img_emb.norm(dim=-1).mean(),rtol=1e-2)
        f_tx, f_y = self.f_net(tx_tensor).mean(), self.f_net(trg_img_emb).mean()
        if self.cfg.optimal_penalty:
            gradient_penalty = gradientOptimality(
                self.f_net, tx_tensor, src_text_cond["text_embed"], self.cfg.coeff_go
            )
        else:
            gradient_penalty = 0.0
        f_loss = f_tx - f_y + gradient_penalty
        log_info = prefix_metrics_keys(
            {
                "f_tx": f_tx,
                "f_y": f_y,
                "gradient_penalty": gradient_penalty,
                "f_tx-f_y": f_tx - f_y,
            },
            "f_loss",
        )
        return f_loss, log_info

    def loss_map(self, src_text_cond, mask=None):
        # src_text_cond = {"text_embed": text_embed, "text_encodings": text_encodings, "unnorm_text_embed": unnorm_text_embed}
        tx_tensor = self.map_t(**src_text_cond)
        cost_loss = self.cost_func(
            src_text_cond["unnorm_text_embed"],
            l2norm(tx_tensor),
            self.cfg.coeff_mse,
            self.cfg.exponent,
        )
        f_tx = self.f_net(tx_tensor).mean()
        map_loss = cost_loss - f_tx
        log_info = prefix_metrics_keys(
            {"cost_loss": cost_loss, "f_tx": f_tx}, "map_loss"
        )
        return map_loss, log_info

    def validation_step(self, batch, batch_idx):
        # evaluate cosine similarity
        trg_img_emb, src_tokens = batch
        text_embed, text_encodings = self.clip.embed_text(src_tokens)
        src_text_cond = {"text_embed": text_embed, "text_encodings": text_encodings}
        self.cos_similarity(src_text_cond, trg_img_emb)

    def cos_similarity(self, src_text_cond, trg_img_emb):
        if self.cfg.ema:
            with self.ema_map.average_parameters():
                tx_tensor = l2norm(self.map_t(**src_text_cond))
        src_txt_emb = src_text_cond["text_embed"]
        txt_trg_sim = -self.cost_func(src_txt_emb, trg_img_emb)
        txt_pf_sim = -self.cost_func(src_txt_emb, tx_tensor)

        pf_trg_sim = -self.cost_func(tx_tensor, trg_img_emb)

        rdm_idx = torch.randperm(trg_img_emb.shape[0])
        unrelated_sim = -self.cost_func(tx_tensor, src_txt_emb[rdm_idx])
        log_info = prefix_metrics_keys(
            {
                "baseline similarity": txt_trg_sim,
                "similarity with text": txt_pf_sim,
                "difference from baseline similarity": abs(txt_trg_sim - txt_pf_sim),
                "similarity with original image": pf_trg_sim,
                "similarity with unrelated caption": unrelated_sim,
            },
            "validation_cos_sim",
        )
        self.log_dict(log_info)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx <= 100:
            # assert 1 == 0, "Too many test samples, terminate earlier."
            if dataloader_idx == 0:
                # visualize the embedding
                trg_img_emb, src_tokens = batch
                text_embed, text_encodings = self.clip.embed_text(src_tokens)
                src_text_cond = {
                    "text_embed": text_embed,
                    "text_encodings": text_encodings,
                }
                pf_img_emb = l2norm(self.map_t(**src_text_cond))

                discrete_ot_map_img_emb = trg_img_emb.detach() * 0
                return pf_img_emb, trg_img_emb, discrete_ot_map_img_emb

            # sampling images
            test_example_data = split_test_full_data(batch, self.device)
            # TODO: this callback can have a problem, we hard code it with index.
            sampling_callback = self.trainer.callbacks[3]
            test_images, test_captions = generate_grid_samples(
                self,
                sampling_callback.decoder,
                sampling_callback.prior,
                test_example_data,
                device=self.device,
                skip_ema=True,
            )
            cherry_pick_img_grid = make_grid(
                test_images, nrow=1, padding=2, pad_value=0
            )
            img_path = f"img_{batch_idx}.png"
            save_image(cherry_pick_img_grid, img_path)
            torch.save(
                {"images": test_images, "captions": test_captions},
                f"raw_data_{batch_idx}.pt",
            )
            # After generating the grid image and saving it
            add_labels_to_image(img_path)

            return None
        return None

    def test_epoch_end(self, outputs):
        for idx, out in enumerate(outputs):
            pf_img_emb, trg_img_emb, discrete_ot_map_img_emb = out
            if idx == 0:
                stacked_pf_feat = pf_img_emb
                stacked_trg_feat = trg_img_emb
                stacked_discrete_ot_feat = torch.from_numpy(discrete_ot_map_img_emb)
            else:
                stacked_pf_feat = torch.cat([stacked_pf_feat, pf_img_emb], dim=0)
                stacked_trg_feat = torch.cat([stacked_trg_feat, trg_img_emb], dim=0)
                stacked_discrete_ot_feat = torch.cat(
                    [
                        stacked_discrete_ot_feat,
                        torch.from_numpy(discrete_ot_map_img_emb),
                    ],
                    dim=0,
                )
        compare_highd_kde_scatter(
            [stacked_pf_feat, stacked_trg_feat, stacked_discrete_ot_feat], "pca.jpg"
        )

    def configure_optimizers(self):
        # These parameters are from LAION pretrained prior.
        optim_map_kwargs = dict(
            lr=self.cfg.lr_T, wd=self.cfg.wd, eps=1e-6, group_wd_params=True
        )

        optimizer_map = get_optimizer(self.map_t.parameters(), **optim_map_kwargs)

        optim_f_kwargs = dict(
            lr=self.cfg.lr_f, wd=self.cfg.wd, eps=1e-6, group_wd_params=True
        )

        optimizer_f = get_optimizer(self.f_net.parameters(), **optim_f_kwargs)
        return optimizer_map, optimizer_f
