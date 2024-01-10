from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding 
from einops import rearrange
 
from dalle_pytorch import distributed_utils
# from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE
from dalle_pytorch.transformer import Transformer, StableLayerNorm

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# discrete vae class

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.,
        normalization = ((0.5,) * 3, (0.5,) * 3)
    ):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = normalization

        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'

        img = self.norm(img)

        logits = self.encoder(img)

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim = -1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out

# main classes

class CLIP(nn.Module):
    def __init__(
        self,
        *,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 10000,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        num_visual_tokens = 512,
        visual_enc_depth = 6,
        visual_heads = 8,
        visual_image_size = 256,
        visual_patch_size = 32,
        channels = 3
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
        self.text_transformer = Transformer(causal = False, seq_len = text_seq_len, dim = dim_text, depth = text_enc_depth, heads = text_heads)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        assert visual_image_size % visual_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (visual_image_size // visual_patch_size) ** 2
        patch_dim = channels * visual_patch_size ** 2

        self.visual_patch_size = visual_patch_size
        self.to_visual_embedding = nn.Linear(patch_dim, dim_image)
        self.visual_pos_emb = nn.Embedding(num_patches, dim_image)
        self.visual_transformer = Transformer(causal = False, seq_len = num_patches, dim = dim_image, depth = visual_enc_depth, heads = visual_heads)
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        self.temperature = nn.Parameter(torch.tensor(1.))

    def forward(
        self,
        text,
        image,
        text_mask = None,
        return_loss = False
    ):
        b, device, p = text.shape[0], text.device, self.visual_patch_size

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        image_patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        image_emb = self.to_visual_embedding(image_patches)
        image_emb += self.visual_pos_emb(torch.arange(image_emb.shape[1], device = device))

        enc_text = self.text_transformer(text_emb, mask = text_mask)
        enc_image = self.visual_transformer(image_emb)

        if exists(text_mask):
            text_latents = masked_mean(enc_text, text_mask, dim = 1)
        else:
            text_latents = enc_text.mean(dim = 1)

        image_latents = enc_image.mean(dim = 1)

        text_latents = self.to_text_latent(text_latents)
        image_latents = self.to_visual_latent(image_latents)

        text_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (text_latents, image_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, image_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', text_latents, image_latents) * temp
        labels = torch.arange(b, device = device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss

# main DALL-E class
@torch.no_grad()
def init_model(m, std=0.02):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if hasattr(m, "bias"):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)


class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        num_text_tokens = 10000,
        text_seq_len = 256,
        depth,
        heads = 8,
        dim_head = 64,
        reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0,
        sparse_attn = False,
        attn_types = None,
        loss_img_weight = 7,
        stable = False
    ):
        super().__init__()
        # assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE)), 'vae must be an instance of DiscreteVAE'

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2

        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)

        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size))

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        # self.vae = vae
        # set_requires_grad(self.vae, False) # freeze VAE from being trained

        self.transformer = Transformer(
            dim = dim,
            causal = True,
            seq_len = seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            reversible = reversible,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = image_fmap_size,
            sparse_attn = sparse_attn,
            stable = stable
        )

        self.stable = stable

        # if stable:
        #     self.norm_by_max = DivideMax(dim = -1)
        self.to_im_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_image_tokens),
        )
        self.to_text_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_text_tokens),
        )

        self.loss_img_weight = loss_img_weight

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        vae,
        text,
        *,
        clip = None,
        mask = None,
        filter_thres = 1.0,
        temperature = 1.,
        img = None,
        num_init_img_tokens = None,
        labels = None,
        cam_mask = None,
        return_img_seq = False
    ):
        text_seq_len, image_seq_len, num_text_tokens = self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        text = text[:, :text_seq_len] # make sure text is within bounds
        out = text

        if exists(img):
            raise NotImplemented
            # image_size = vae.image_size
            # assert img.shape[1] == 3 and img.shape[2] == image_size and img.shape[3] == image_size, f'input image must have the correct image size {image_size}'
            #
            # indices = vae.get_codebook_indices(img)
            # num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))  # OpenAI used 14 * 32 initial tokens to prime
            # assert num_img_tokens < image_seq_len, 'number of initial image tokens for priming must be less than the total image token sequence length'
            #
            # indices = indices[:, :num_img_tokens]
            # out = torch.cat((out, indices), dim = -1)

        assert out.size(-1) == text_seq_len
        for cur_len in range(out.shape[1], total_len):
            # is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self(text, image, mask = mask)[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)

            # sample -= (num_text_tokens if is_image else 0)
            # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample), dim=-1)
            if labels is not None:
                # import pdb; pdb.set_trace()
                out[~cam_mask[:, cur_len-text_seq_len], -1] = labels[~cam_mask[:, cur_len-text_seq_len], cur_len-text_seq_len]

            if out.shape[1] <= text_seq_len:
                mask = F.pad(mask, (0, 1), value = True)

        text_seq = out[:, :text_seq_len]

        img_seq = out[:, -image_seq_len:]

        images = vae(img_seq)

        if exists(clip):
            scores = clip(text_seq, images, return_loss = False)
            return images, scores

        if return_img_seq:
            return images, img_seq
        else:
            return images

    def forward(
        self,
        text,
        image = None,
        mask = None,
        return_loss = False
    ):
        assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
        device, total_seq_len = text.device, self.total_seq_len

        meaning_text_token_mask_0pad = torch.zeros_like(text)
        meaning_text_token_mask_0pad[torch.nonzero(text, as_tuple=True)] = 1

        # make sure padding in text tokens get unique padding token id
        text_range = torch.arange(self.text_seq_len, device = device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        # add <bos>
        text = F.pad(text, (1, 0), value = 0)

        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        seq_len = tokens.shape[1]
        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4

            assert not is_raw_image
            image_len = image.shape[1]
            image_emb = self.image_emb(image)

            image_emb += self.image_pos_emb(image_emb)

            tokens = torch.cat((tokens, image_emb), dim = 1)

            seq_len += image_len
        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained

        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        out = self.transformer(tokens)
        # if self.stable:
        #     out = self.norm_by_max(out)

        text_out = out[:, :self.text_seq_len, :]
        img_out = out[:, self.text_seq_len:, :]
        text_logits = self.to_text_logits(text_out)
        img_logits = self.to_im_logits(img_out)
        # mask logits to make sure text predicts text (except last token), and image predicts image

        if not return_loss:
            # if is_empty(img_logits):
            #     return text_logits  # use the last text token to predict the first image token
            # else:
            return img_logits

        assert exists(image), 'when training, image must be supplied'

        text_logits = rearrange(text_logits, 'b n c -> b c n')
        img_logits = rearrange(img_logits, 'b n c -> b c n')

        loss_text = F.cross_entropy(text_logits, text[:, 1:])
        loss_img = F.cross_entropy(img_logits, image)

        loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)

        loss_meaningful_text = F.cross_entropy(text_logits, text[:, 1:], reduction='none').detach()
        loss_meaningful_text = (loss_meaningful_text * meaning_text_token_mask_0pad).sum() / meaning_text_token_mask_0pad.sum()

        return loss, (loss_text, loss_img), loss_meaningful_text, rearrange(img_logits, 'b c n -> b n c')



class DALLE_CONTEXT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        num_text_tokens = 10000,
        text_seq_len = 256,
        depth,
        heads = 8,
        dim_head = 64,
        reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0,
        sparse_attn = False,
        attn_types = None,
        loss_img_weight = 7,
        loss_vc_weight = 1,
        stable = False,
    ):
        super().__init__()
        # assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE)), 'vae must be an instance of DiscreteVAE'

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2

        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)

        self.text_emb = nn.Embedding(num_text_tokens, dim)
        self.image_emb = nn.Embedding(num_image_tokens, dim)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size))

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        # self.vae = vae
        # set_requires_grad(self.vae, False) # freeze VAE from being trained

        self.bov_emb = nn.Parameter(torch.randn(1, 1, dim))
        self.bov_pos_emb = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(
            dim = dim,
            causal = True,
            seq_len = seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            reversible = reversible,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = image_fmap_size,
            sparse_attn = sparse_attn,
            stable = stable
        )

        self.stable = stable

        # if stable:
        #     self.norm_by_max = DivideMax(dim = -1)
        # self.bos_token_id_in_vc_logits = num_image_tokens
        # self.to_vc_logits = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_image_tokens),
        # )
        self.to_im_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_image_tokens),
        )
        self.to_text_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_text_tokens),
        )

        self.loss_img_weight = loss_img_weight
        self.loss_vc_weight = loss_vc_weight

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        vae,
        text,
        *,
        visual_context=None,
        context_mask_0pad=None,
        transformer_attn_kwargs={},
        clip = None,
        mask = None,
        filter_thres = 1.0,
        temperature = 1.,
        img = None,
        num_init_img_tokens = None,
        labels = None,
        cam_mask = None,
        return_img_seq = False
    ):
        text_seq_len, image_seq_len, num_text_tokens = self.text_seq_len, self.image_seq_len, self.num_text_tokens
        cat_out_total_len = text_seq_len + image_seq_len

        assert text.shape[1] == self.text_seq_len  # make sure text is within bounds
        text_image_concat_out = text

        if exists(visual_context):
            visual_context_len = visual_context.shape[1]
            # cat_out_total_len += (visual_context_len + 1)

        # assert out.size(-1) == text_seq_len

        for cur_len in range(text_image_concat_out.shape[1], cat_out_total_len):
            # is_image = cur_len >= text_seq_len

            text, image = text_image_concat_out[:, :text_seq_len], text_image_concat_out[:, text_seq_len:]

            if len(transformer_attn_kwargs) > 0:
                # total_seq_len = transformer_attn_kwargs["mask"].size(-1)
                # import pdb; pdb.set_trace()
                cur_transformer_attn_kwargs = \
                    {"mask": transformer_attn_kwargs["mask"][:,
                             :(1+visual_context_len+1+text_seq_len+image.shape[-1]),
                             :(1+visual_context_len+1+text_seq_len+image.shape[-1])]}
            else:
                cur_transformer_attn_kwargs = transformer_attn_kwargs

            logits = self(
                text, image,
                visual_context=visual_context,
                context_mask_0pad=context_mask_0pad,
                transformer_attn_kwargs=cur_transformer_attn_kwargs
            )[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)

            # sample -= (num_text_tokens if is_image else 0)
            # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            text_image_concat_out = torch.cat((text_image_concat_out, sample), dim=-1)
            if labels is not None:
                # import pdb; pdb.set_trace()
                text_image_concat_out[~cam_mask[:, cur_len-text_seq_len], -1] = labels[~cam_mask[:, cur_len-text_seq_len], cur_len-text_seq_len]

            if text_image_concat_out.shape[1] <= text_seq_len:
                mask = F.pad(mask, (0, 1), value = True)
                raise NotImplemented

        text_seq = text_image_concat_out[:, :text_seq_len]

        img_seq = text_image_concat_out[:, -image_seq_len:]

        images = vae(img_seq)

        if exists(clip):
            scores = clip(text_seq, images, return_loss = False)
            return images, scores

        if return_img_seq:
            return images, img_seq
        else:
            return images

    def forward(
        self,
        text,
        image = None,
        visual_context = None,
        context_mask_0pad = None,
        transformer_attn_kwargs = {},
        return_loss = False
    ):
        #
        assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
        device = text.device

        meaning_text_token_mask_0pad = torch.zeros_like(text)
        meaning_text_token_mask_0pad[torch.nonzero(text, as_tuple=True)] = 1

        # make sure padding in text tokens get unique padding token id
        text_range = torch.arange(self.text_seq_len, device = device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        # add <bos>
        text = F.pad(text, (1, 0), value = 0)

        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device = device))
        total_seq_len = self.text_seq_len + self.image_seq_len

        # Visual Context Token
        if exists(visual_context) and not is_empty(visual_context):
            is_raw_image = len(visual_context.shape) == 4
            assert not is_raw_image
            visual_context_len = visual_context.shape[1]

            visual_context_emb = self.image_emb(visual_context)  # (B, T, C)
            visual_context_emb += self.image_pos_emb(visual_context_emb)  # (B, T, C)

            bov_token_emb = self.bov_emb + self.bov_pos_emb
            visual_context_emb = torch.cat([
                bov_token_emb.repeat(visual_context_emb.size(0), 1, 1),
                visual_context_emb
            ], dim=1)  # (B, T+1, C)
            tokens = torch.cat((visual_context_emb, tokens), dim=1)

            # vc_logits_labels = F.pad(visual_context, pad=(0, 1), value=self.bos_token_id_in_vc_logits)
            total_seq_len = total_seq_len + visual_context_len + 1
        else:
            visual_context_len = 0

        seq_len = tokens.shape[1]

        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4
            assert not is_raw_image
            image_len = image.shape[1]
            image_emb = self.image_emb(image)

            image_emb += self.image_pos_emb(image_emb)

            tokens = torch.cat((tokens, image_emb), dim = 1)

            seq_len += image_len

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained
        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        out = self.transformer(tokens, **transformer_attn_kwargs)
        # if self.stable:
        #     out = self.norm_by_max(out)

        if visual_context_len == 0:
            vc_out = out[:, 0:0, :]
            text_out = out[:, 0:0 + self.text_seq_len, :]
            img_out = out[:, 0 + self.text_seq_len:, :]
        else:
            vc_out = out[:, :visual_context_len, :]
            text_out = out[:, visual_context_len:visual_context_len+self.text_seq_len+1, :]
            img_out = out[:, visual_context_len+self.text_seq_len+1:, :]
        vc_logits = self.to_im_logits(vc_out)
        text_logits = self.to_text_logits(text_out)
        img_logits = self.to_im_logits(img_out)
        # mask logits to make sure text predicts text (except last token), and image predicts image

        if not return_loss:
            # if is_empty(img_logits):
            #     return text_logits  # use the last text token to predict the first image token
            # else:
            # import pdb; pdb.set_trace()
            return img_logits

        assert exists(image), 'when training, image must be supplied'

        vc_logits = rearrange(vc_logits, 'b n c -> b c n')
        text_logits = rearrange(text_logits, 'b n c -> b c n')
        img_logits = rearrange(img_logits, 'b n c -> b c n')

        loss_img = F.cross_entropy(img_logits, image)
        # loss computation for visual context and text
        if not exists(visual_context) or context_mask_0pad.sum() == 0:
            loss_vc = torch.zeros(1).to(loss_img)
        else:
            loss_vc = F.cross_entropy(vc_logits, visual_context, reduction='none')  # (b, n)
            loss_vc = (loss_vc * context_mask_0pad.unsqueeze(1).float()).sum() / \
                      (context_mask_0pad.sum(0) * loss_vc.size(1)).float()

        loss_text_mask_0pad = torch.ones_like(text).float()
        if len(transformer_attn_kwargs) == 0:
            loss_text_mask_0pad[~context_mask_0pad, 0] = 0.
        else:
            loss_text_mask_0pad[:, 0] = 0.
        loss_text = F.cross_entropy(text_logits, text, reduction='none')
        loss_text = (loss_text * loss_text_mask_0pad).sum() / loss_text_mask_0pad.sum()

        loss = (loss_text + self.loss_vc_weight * loss_vc + self.loss_img_weight * loss_img) / \
               (self.loss_img_weight + self.loss_vc_weight + 1)

        # loss_meaningful_text = F.cross_entropy(text_logits[:, :, 1:], text[:, 1:], reduction='none').detach()
        # loss_meaningful_text = (loss_meaningful_text * meaning_text_token_mask_0pad).sum() / meaning_text_token_mask_0pad.sum()

        return loss, (loss_text, loss_vc, loss_img), torch.zeros(1).to(loss), rearrange(img_logits, 'b c n -> b n c')
