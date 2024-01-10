"""
CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP]
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import CLIP

from .. import EXTRCT_REGISTRY

class ClipExtractorBase(nn.Module):
    def __init__(
        self, 
        model_name: str = 'ViT-B/32',
        **kwargs) -> None:
        super().__init__()

        clip_state_dict = CLIP.get_config(model_name=model_name)
        self.clip, clip_embed_dim = self.load_config(clip_state_dict)
        self.feature_dim = clip_embed_dim
        self.dtype = self.clip.dtype

    def load_config(self, clip_state_dict):

        # set the parameters of CLIP
        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = 49408
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        cut_top_layer = 0
        
        model = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
        ).float()

        ret = model.load_state_dict(clip_state_dict, strict=False)
        return model, embed_dim

@EXTRCT_REGISTRY.register()
class ClipVideoExtractor(ClipExtractorBase):
    
    def __init__(self, model_name: str = 'ViT-B/32', **kwargs) -> None:
        super().__init__(model_name=model_name)
        self.model = self.clip.visual
        self.clip = None

    def mean_pooling(self, visual_output, video_mask):
        """average pooling for the overall video representation
        Args:
            visual_output: embedding
            video_mask: video embedding
        Returns:
            video_out: output embedding [1,512]
        """
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def forward(self, batch):
        """video encoder
        Returns:
            x: output embedding [1,512]
        """

        video, video_mask = batch['videos'], batch['video_masks']
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        bs, ts, channel, h, w = video.shape
        video = video.view(bs * ts, channel, h, w)
        video_frame = bs * ts

        hidden = self.model(video.type(self.dtype), video_frame=video_frame)
        hidden = self.model.ln_post(hidden) @ self.model.proj
        visual_hidden = hidden[:, 0, :]
        visual_hidden = visual_hidden.view(bs, -1, visual_hidden.size(-1))

        # pooling
        pooled_output = self.mean_pooling(visual_hidden, batch['video_masks'])

        return pooled_output

@EXTRCT_REGISTRY.register()
class ClipImageExtractor(ClipExtractorBase):
    
    def __init__(self, model_name: str = 'ViT-B/32', **kwargs) -> None:
        super().__init__(model_name=model_name)
        self.model = self.clip.visual
        self.clip = None
        
    def forward(self, x):
        """video encoder
        Returns:
            x: output embedding [1,512]
        """
        hidden = self.model(x.type(self.dtype), video_frame=x.shape[0])
        hidden = self.model.ln_post(hidden) @ self.model.proj
        x = hidden[:, 0, :]
        return x

@EXTRCT_REGISTRY.register()
class ClipTextExtractor(ClipExtractorBase):
    
    def __init__(self, model_name: str = 'ViT-B/32', **kwargs) -> None:
        super().__init__(model_name=model_name)
        self.model = self.clip.visual
        self.token_embedding = self.clip.token_embedding
        self.positional_embedding = self.clip.positional_embedding
        self.transformer = self.clip.transformer
        self.text_projection = self.clip.text_projection
        self.ln_final = self.clip.ln_final
        self.clip = None

    def forward(self, batch):
        """text encoder
        Args:
            text: caption
            return_hidden: whether to return hidden variable
        Returns:
            x: output embedding [1,512]
        """
        x = self.token_embedding(batch["input_ids"]).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), batch["input_ids"].argmax(dim=-1)]

        return x