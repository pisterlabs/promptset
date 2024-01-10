import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from mmvid_pytorch.modules import AxialPositionalEmbeddingList
from utils.utils import DivideMax

FAKE_POOL_SIZE = 64


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


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


def set_requires_grad(model, value):
    if model is not None:
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


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# sampling helpers


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, idx = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, idx, val)
    return probs


# Augmentation helpers
from itertools import permutations

PERM_LIST = None


def randperm(n, ordered=False):
    global PERM_LIST
    if ordered:
        return torch.randperm(n)
    else:
        if n < 6:
            if PERM_LIST is None:
                PERM_LIST = list(permutations(range(n)))[1:]  # permutation returns n!
            return random.choice(PERM_LIST)
        perm_ord = torch.tensor(range(n))
        while True:
            perm = torch.randperm(n)
            if (perm != perm_ord).any():
                return perm


def swap(tensor, dim=0):
    """
    if the input dimension is even, then split the tensor in the middle and then swap and concatenate,
    otherwise randomly permute the dimension
    :param tensor:
    :param dim:
    :return: swapped tensor with the same shape
    """
    if tensor.shape[dim] % 2 == 0:
        tensor_swapped = torch.cat(torch.chunk(tensor, 2, dim=dim)[::-1], dim=dim)  # split in the middle and swap
    else:
        idx_perm = randperm(tensor.shape[dim], False)
        if dim == 0:
            tensor_swapped = tensor[idx_perm, ...]
        elif dim == 1:
            tensor_swapped = tensor[:, idx_perm, ...]
        else:
            raise RuntimeError
    return tensor_swapped


def swap_one_frame_along_batch(tokens, t=1, shuffle=False):
    tokens_copy = tokens.detach().clone()
    b, n, c = tokens.shape
    tokens_shuffled = tokens_copy.reshape(b, t, n // t, c)
    idx = np.random.randint(0, t, b)  # get b random integers
    if shuffle:
        perm_idx = randperm(t)
        frames_shuffled = tokens_shuffled[range(b), idx, ...][perm_idx, ...]
    else:
        frames_shuffled = swap(tokens_shuffled[range(b), idx, ...], 0)
    tokens_shuffled[range(b), idx, ...] = frames_shuffled
    tokens_shuffled = tokens_shuffled.reshape(b, n, c)
    return tokens_shuffled


# Color Jittering in Negative Video Augmentation
def warp_video_with_color(video):
    # video (B, T, 3, H, W)
    out = []
    for n in range(video.shape[0]):
        x = video[n]  # (T, 3, H, W)
        c_shift = torch.rand(1, device=x.device) - 0.5
        m = torch.zeros_like(x)  # (T, 3, H, W)
        num = random.randint(0, 3)
        if num == 0:
            m.data += c_shift
        elif num == 1:
            m[:, 0].data += c_shift
        elif num == 2:
            m[:, 1].data += c_shift
        else:
            m[:, 2].data += c_shift
        out.append(torch.clamp(x + m, 0, 1))
    return torch.stack(out)


def warp_with_color(x):
    # x: (3, h, w)
    c_shift = torch.rand(1, device=x.device) - 0.5
    m = torch.zeros_like(x)
    num = random.randint(0, 3)
    if num == 0:
        m.data += c_shift
    elif num == 1:
        m[0].data += c_shift
    elif num == 2:
        m[1].data += c_shift
    else:
        m[2].data += c_shift
    out = torch.clamp(x + m, 0, 1)
    return out.unsqueeze(0)  # out: (1, 3, h, w)


# Affine Transform in Negative Video Augmentation
def warp_with_affine(x, angle=180, trans=0.1, scale=0.05):
    """
    Applying affine transformation to one frame
    :param x: input frame: (3, H, W)
    :param angle:
    :param trans:
    :param scale:
    :return:
    """
    angle = np.pi * angle / 180.

    pa = torch.FloatTensor(4)
    th = torch.FloatTensor(2, 3)

    pa[0].uniform_(-angle, angle)
    pa[1].uniform_(-trans, trans)
    pa[2].uniform_(-trans, trans)
    pa[3].uniform_(1. - scale, 1. + scale)

    th[0][0] = pa[3] * torch.cos(pa[0])
    th[0][1] = pa[3] * torch.sin(-pa[0])
    th[0][2] = pa[1]
    th[1][0] = pa[3] * torch.sin(pa[0])
    th[1][1] = pa[3] * torch.cos(pa[0])
    th[1][2] = pa[2]

    x = x.unsqueeze(0)
    th = th.unsqueeze(0)
    grid = F.affine_grid(th, x.size()).to(x.device)
    out = F.grid_sample(x, grid, padding_mode="reflection")
    return out[0]  # out (3, h, w)


def warp(x, vid_strategy_prob=None):
    """
    negative video augmentation
    :param x: input frame sequences: (b, t, c, h, w)
    :param vid_strategy_prob:
    :return: out: (b, t, c, h, w)
    """
    if vid_strategy_prob is None:
        vid_strategy_prob = [0.25, 0.25, 0.25, 0.25]
    b, t, c, h, w = x.shape
    out = []
    num_strategy = len(vid_strategy_prob)
    assert num_strategy == 4
    for i in range(b):
        strategy = np.random.choice(range(num_strategy), p=vid_strategy_prob)
        if strategy == 0:
            # frame swapping: a random frame is replaced by using a frame from another video
            idx_expt_i = np.random.choice(list(set(range(b)) - {i}))
            y = x[i].detach().clone()
            j1 = random.randint(0, t - 1)
            j2 = random.randint(0, t - 1)
            y[j1, ...] = x[idx_expt_i, j2, ...]
            out.append(y)
        elif strategy == 1:
            # frame shuffling: shuffle frames within a sequence
            perm_idx = randperm(t)
            y = x[i, perm_idx, ...].detach().clone()
            out.append(y)
        elif strategy == 2:
            # color jittering
            j1 = random.randint(0, t - 1)
            y = x[i].detach().clone()
            y[j1, ...] = warp_with_color(y[j1]).squeeze(0)
            out.append(y)
        elif strategy == 3:
            # affine transform
            j1 = random.randint(0, t - 1)
            y = x[i].detach().clone()
            y[j1, ...] = warp_with_affine(y[j1], 30, 0.1, 0.1)
            out.append(y)
        else:
            raise NotImplementedError
    out = torch.stack(out, 0)
    return out


# discrete vae class
class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channel, channel, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(channel, channel, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(channel, channel, 1))

    def forward(self, x):
        return self.net(x) + x


# DALL-E class
class BERT(nn.Module):
    def __init__(
            self,
            *,
            dim,
            vae,
            cvae=None,
            num_text_tokens=10000,
            text_seq_len=256,
            stable=False,
            text_feature_dim=0,
            fixed_language_model=None,
            which_transformer='none',
            num_visuals=1,
            num_targets=1,
            use_separate_visual_emb=False,
            insert_sep=False,
            text_emb_bottleneck=False,
            **kwargs,
    ):
        super().__init__()
        """
        Special Tokens:
        [REL]  if text-video are relevant
        [VID]  if video is continuous (shuffle frames)
        [MASK] masking
        [SEP]  separation (reserved)
        """
        # get evaluation mode
        self.eval_mode = kwargs['eval_mode']

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2
        self.dim = dim

        self.num_visuals = num_visuals
        self.num_targets = num_targets

        # For block masking
        self.random_erasing = T.RandomErasing(p=1, scale=(0.2, 0.8), ratio=(0.5, 2), value=0)  # erase visual

        if fixed_language_model is None:
            # extend the length of the text embeddings
            # reserve unique padding tokens for each position (text sequence length)
            num_text_tokens = num_text_tokens + text_seq_len
            self.text_emb = nn.Embedding(num_text_tokens, dim)
            self.text_pos_emb = nn.Embedding(text_seq_len, dim)
            self.text_feature_mapping = lambda x: x
        else:
            assert text_feature_dim > 0
            text_seq_len = 1  # NOTE: if use fixed language model, text_seq_len is 1
            num_text_tokens = 1
            self.text_emb, self.text_pos_emb = None, None
            if text_emb_bottleneck is not None:
                nf = int(text_emb_bottleneck)
                self.text_feature_mapping = nn.Sequential(
                    nn.LayerNorm(text_feature_dim),
                    nn.Linear(text_feature_dim, nf),
                    nn.LayerNorm(nf),
                    nn.Linear(nf, dim),
                    nn.LayerNorm(dim),
                )
            else:
                self.text_feature_mapping = nn.Linear(text_feature_dim, dim)

        # TODO: for masking+separate visual
        self.image_emb = nn.Embedding(num_image_tokens + 2, dim)
        self.target_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(num_targets, image_fmap_size, image_fmap_size))

        if cvae is not None:
            use_separate_visual_emb = True
        if num_visuals > 0:
            if use_separate_visual_emb:
                # TODO: for masking+separate visual
                self.visual_emb = nn.Embedding(num_image_tokens + 2, dim)
            else:
                self.visual_emb = None

            self.visual_pos_emb = AxialPositionalEmbeddingList(
                dim,
                num_visuals,
                axial_shape=(image_fmap_size, image_fmap_size))

        # [0, num_image_tokens-1] is the index for image tokens
        self.image_token_lut = {
            '[MASK]': num_image_tokens,
            '[SEP]': num_image_tokens + 1,
        }
        # for offsetting logits index and calculating cross entropy loss
        self.num_text_tokens = num_text_tokens
        self.num_image_tokens = num_image_tokens
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len
        self.image_fmap_size = image_fmap_size
        self.image_size = image_size
        self.visual_seq_len = num_visuals * (image_seq_len + insert_sep)
        self.target_seq_len = num_targets * image_seq_len
        self.insert_sep = insert_sep

        self.special_token_lut = {
            '[REL]': 0,
            # ---------- before and after control sequence ----------
            '[ST1]': 1,
            '[VID]': 2,
            '[ST3]': 3,
            '[ST4]': 4,
        }  # NOTE: [ST{1,3,4}] are reserved for future use
        self.num_special_tokens = len(self.special_token_lut)
        self.special_emb = nn.Embedding(self.num_special_tokens, dim)
        self.special_pos_emb = nn.Embedding(self.num_special_tokens, dim)

        self.before_control_tok = [0]  # REL
        self.after_control_tok = [1, 2]  # ST1, VID
        self.before_control_seq_len = len(self.before_control_tok)
        self.after_control_seq_len = len(self.after_control_tok)

        self.rel_tok_index = 0  # always starts from 0
        self.txt_tok_index = self.before_control_seq_len
        self.st1_tok_index = self.before_control_seq_len + self.text_seq_len + self.visual_seq_len
        self.vid_tok_index = self.before_control_seq_len + self.text_seq_len + self.visual_seq_len + 1


        # for text-vid generation, this value is: 1+text_seq_len+2+8*image_seq_len
        self.total_seq_len = self.before_control_seq_len + \
                             self.text_seq_len + \
                             self.visual_seq_len + \
                             self.after_control_seq_len + \
                             self.target_seq_len

        self.vae = vae
        self.cvae = cvae
        set_requires_grad(self.vae, False)  # freeze VAE from being trained
        set_requires_grad(self.cvae, False)  # freeze cVAE from being trained

        self.fixed_language_model = fixed_language_model
        self.which_transformer = which_transformer
        mask_prev_index = [self.st1_tok_index, self.vid_tok_index]
        assert which_transformer != 'default', "using the default transformer model!"
        if which_transformer.startswith('openai_clip'):
            from mmvid_pytorch.transformers.clip_model import OpenAICLIPTransformer
            self.transformer = OpenAICLIPTransformer(
                self.total_seq_len,
                which_transformer,
                model_path=kwargs['openai_clip_path'],
                causal=True,
                mask_type='mask_prev',
                mask_kwargs={'index': mask_prev_index},
            )
        else:  # NOTE: You can port the Transformer from dalle_pytorch if you want to train from scratch
            raise NotImplementedError

        self.stable = stable

        if stable:  # max normalization
            self.norm_by_max = DivideMax(dim=-1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_image_tokens),
        )
        # binary classification for REL token
        self.to_logits_rel = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        # binary classification for VID token
        self.to_logits_vid = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

        self.current_step = 0
        # erase visual
        self.visual_eraser = T.RandomErasing(p=0.95,
                                             scale=(0.55, 0.85),
                                             ratio=(0.5, 2),
                                             value=self.num_image_tokens)

    @torch.no_grad()
    @eval_decorator
    def generate_images(
            self,
            text,
            *,
            visual=None,
            mask=None,
            img=None,
            argmax=False,
            dynamic=True,
            debug=False,
            erase_visual=False,
            mask_predict_steps=10,
            preserve=None,
            t_overlap=1,
            pc_mode=None,
            vc_mode=None,
            face_mode=None,
            mp_config=None,
    ):
        control_emb = self(
            text,
            visual=visual,
            erase_visual=erase_visual,
            erase_visual_half=True,  # NOTE: always erase half during generation if erase visual
            vc_mode=vc_mode,
            face_mode=face_mode,
            return_loss=False,
        )
        img_seq, pnag_samples = self.mask_predict(
            control_emb,
            argmax=argmax,
            dynamic=dynamic,
            debug=debug,
            steps=mask_predict_steps,
            preserve=preserve,
            t_overlap=t_overlap,
            pc_mode=pc_mode,
            mp_config=mp_config,
        )
        img_seq = rearrange(img_seq, 'b (t n) -> (b t) n', n=self.image_seq_len)  # image token
        images = self.vae.decode(img_seq)
        images = rearrange(images, '(b t) c h w -> b t c h w', t=self.num_targets)
        return images, pnag_samples, img_seq

    def transformer_forward(self, tokens):
        # tokens are embeddings
        out = self.transformer(tokens)
        if self.stable:
            out = self.norm_by_max(out)
        return out

    def decode_images(self, img_seq):
        img_seq = rearrange(img_seq, 'b (t n) -> (b t) n', n=self.image_seq_len)
        images = self.vae.decode(img_seq)
        return images

    def decode_masks(self, mask):
        mask = rearrange(mask, 'b (t h w) -> (b t) 1 h w',
                         h=self.image_fmap_size,
                         w=self.image_fmap_size)
        patch_size = self.image_size // self.image_fmap_size
        # make the mask size the same as the original input image
        mask_ = torch.repeat_interleave(torch.repeat_interleave(mask, patch_size, 2), patch_size, 3)
        mask = F.pad(mask_, (0, 0, 0, 0, 0, 2))  # (pad_left, pad_right, pad_top, pad_bot, pad_front, pad_back)
        return mask

    @torch.no_grad()
    def mask_predict(
            self,
            control_emb,
            dynamic=True,
            debug=False,
            steps=10,
            preserve=None,
            t_overlap=1,
            mp_config=None,
            **kwargs,
    ):
        def log(logit, eps=1e-20):
            return torch.log(logit + eps)

        def gumbel_noise(logit):
            noise = torch.rand_like(logit)
            return -log(-log(noise))

        def sample_gumbel(logit, eps=1e-20):
            U = torch.rand_like(logit)
            return -torch.log(-torch.log(U + eps) + eps)

        def sample_multinomial(logits, temperature=1.):
            logits = logits + max(temperature, 1e-10) * gumbel_noise(logits)
            probs = F.softmax(logits, dim=2)
            tok = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)
            tok = rearrange(tok, '(b n) 1 -> b n 1', b=probs.shape[0])
            Y = torch.gather(probs, 2, tok)
            Y, tok = Y.squeeze(2), tok.squeeze(2)
            return Y, tok

        # control_emb: [B, 1 + text_seq_len + 2, dim] or [B, 1 + text_seq_len + visual_seq_len + 2, dim]
        control_seq_len, device = control_emb.shape[1], control_emb.device
        # N is number of masked sequence
        if self.eval_mode == 'phenaki':
            if preserve is None:
                t_overlap = 0
            N = self.target_seq_len - self.image_seq_len * t_overlap
        # elif long_mode == 'interp' or long_mode == 'interp2' or long_mode == 'interp_real':
        #     N = self.target_seq_len // 2
        else:
            N = self.target_seq_len

        bs = control_emb.shape[0]
        # print(self.target_seq_len) 512
        fully_masked_tok = torch.full((1, self.target_seq_len), self.image_token_lut['[MASK]'], device=device).long()
        preserve_mask1 = torch.zeros((1, self.target_seq_len), device=device).long()
        preserve_ = torch.full((bs, self.target_seq_len), self.image_token_lut['[MASK]'], device=device).long()

        has_prime_frame = False
        if preserve is not None:
            has_prime_frame = True if t_overlap != 0 else False
            if self.eval_mode == 'phenaki':
                print("overlapping t is: ", t_overlap)
                preserve_mask1[:, :self.image_seq_len * t_overlap] = 1
                preserve = rearrange(preserve, '(b t) n -> b (t n)', b=bs, n=self.image_seq_len)  # [bs, 512]
                preserve_[:, :self.image_seq_len * t_overlap] = preserve[:, -self.image_seq_len * t_overlap:]
            # elif long_mode == 'interp' or long_mode == 'interp2' or long_mode == 'interp_real':
            #     preserve_mask1 = rearrange(preserve_mask1, 'b (t n) -> b t n', t=self.num_targets)
            #     preserve_mask1[:, ::2, :] = 1
            #     preserve = rearrange(preserve, 'b (t n) -> b t n', t=self.num_targets)
            #     preserve_ = rearrange(preserve_, 'b (t n) -> b t n', t=self.num_targets)
            #     preserve_[:, ::2, :] = preserve[:, :self.num_targets // 2, :]
            #     preserve_ = rearrange(preserve_, 'b t n -> b (t n)')
            #     preserve_mask1 = rearrange(preserve_mask1, 'b t n -> b (t n)')

        preserve = preserve_
        del preserve_
        preserve_mask1 = preserve_mask1.bool()

        # @Crane: modify
        # fully_masked_emb = self.image_emb(preserve)
        # target_pos_emb = self.target_pos_emb(fully_masked_emb)

        fully_masked_emb = self.image_emb(fully_masked_tok)
        target_pos_emb = self.target_pos_emb(fully_masked_emb)
        mask_emb = self.image_emb.weight[self.image_token_lut['[MASK]']]

        # NOTE: steps can overwrite T in mp_config if positive
        Tmax = mp_config['T'] if steps <= 0 else steps
        B = mp_config['B']  # beam-search parameter

        # Eq (4)
        T1_n = mp_config['T1_n']  # L1
        T2_n = mp_config['T2_n']  # L2
        T3_n = mp_config['T3_n']  # L3
        N1_n = mp_config['N1_n']  # alpha1
        N2_n = mp_config['N2_n']  # beta1
        N3_n = max(1, int(N * mp_config['N3_n']))  # alpha2
        N4_n = max(1, int(N * mp_config['N4_n']))  # alpha3
        n = list(N * np.linspace(N1_n, N2_n, T1_n)) + list(N3_n * np.ones(T2_n)) + list(N4_n * np.ones(T3_n))

        # Eq (5)
        T1_t = mp_config['T1_t']  # L1
        T2_t = mp_config['T2_t']  # L2
        T3_t = mp_config['T3_t']  # L3
        N1_t = mp_config['N1_t']  # alpha1
        N2_t = mp_config['N2_t']  # beta1
        N3_t = mp_config['N3_t']  # alpha2
        N4_t = mp_config['N4_t']  # alpha3
        temp = list(np.linspace(N1_t, N2_t, T1_t)) + list(N3_t * np.ones(T2_t)) + list(N4_t * np.ones(T3_t))

        # take the integer of # of masked tokens
        n = list(map(int, n))

        image_samples = []
        sample_toks = []
        for i in range(bs):
            control_emb_ = control_emb[i:i + 1]  # [53=1 + text_seq_len + 2, C=768]

            tok_in = fully_masked_tok  # [1, target_seq_len]
            if has_prime_frame:
                tok_in[0] = torch.where(preserve_mask1, preserve[i], fully_masked_tok[0])

            emb_in = self.image_emb(tok_in)
            # add
            target_pos_emb = self.target_pos_emb(emb_in)
            tokens = torch.cat((control_emb_, emb_in + target_pos_emb), dim=1)  # z_in
            out = self.transformer_forward(tokens)[:, control_seq_len:, :]
            logits = self.to_logits(out)  # p_tilde: (B, N, C)
            # line 2: sample token
            Y, I_new = sample_multinomial(logits, temp[0])  # Y: [B, 512]
            # line 3
            I_tok = torch.where(preserve_mask1, preserve[i:i + 1], I_new)  # z_out

            if debug:
                print('PNAG:')
                image_samples.append(self.decode_images(I_tok))

            # Mask-Predict
            Smax = 0
            tmax = 0
            Imax = None
            # print(f"beam search size is: {B}")
            for t in range(1, Tmax):
                # Mask: sample B seqs [I_in] (masked sequences) according to Y
                emb_in = []
                masks1 = []
                # print(f"current step is: {t}")
                for j in range(B):
                    Y_valid = Y[~preserve_mask1]  # Line 1 in A3: collect probabilities no need to preserve
                    idx_valid = torch.arange(self.target_seq_len, device=device)[~preserve_mask1[0]]  # Line 2 in A3
                    try:
                        mask1_idx = torch.multinomial(Y_valid, N - n[t - 1], replacement=False)  # Line 3 in A3
                    except RuntimeError:
                        mask1_idx = torch.multinomial(Y_valid, 1, replacement=False)
                    mask1_idx = idx_valid[mask1_idx]  # Line 4 in A3
                    # Line 5 in A3
                    mask1 = torch.zeros(self.target_seq_len, device=device).scatter_(0, mask1_idx, 1).unsqueeze(0)
                    # ---- Line 6 in A3 ---- #
                    mask1 = torch.logical_or(mask1, preserve_mask1)
                    masks1.append(mask1)
                    # ---- end ---- #
                    emb_out = self.image_emb(I_tok)
                    emb_masked = torch.where(mask1.unsqueeze(2), emb_out, mask_emb)
                    emb_in.append(emb_masked)

                # Predict: predict I_out and select b with the highest score; update Y
                S = torch.zeros(B)
                S_rel = torch.zeros(B)
                S_vid = torch.zeros(B)
                YB, tokB = [], []
                for j in range(B):
                    # re-predict
                    tokens = torch.cat((control_emb_, emb_in[j] + target_pos_emb), dim=1)  # z_in
                    out = self.transformer_forward(tokens)
                    logits = self.to_logits(out[:, control_seq_len:, :])  # p^{tilde}^{b}: (B, N, C)
                    # Line 11: sample token
                    Y_new, I_new = sample_multinomial(logits, temp[t])
                    # Line 6 in A3
                    mask1_j = torch.logical_or(masks1[j], preserve_mask1)
                    # Line 7: remask Y and  tokens
                    Y = torch.where(mask1_j, Y, Y_new)
                    I_tok = torch.where(mask1_j, I_tok, I_new)

                    S_rel[j] = F.sigmoid(self.to_logits_rel(out[:, self.rel_tok_index, :]))
                    S_vid[j] = F.sigmoid(self.to_logits_vid(out[:, self.vid_tok_index, :]))
                    S[j] = S_rel[j] * 0.5 + S_vid[j] * 0.5
                    YB.append(Y)
                    tokB.append(I_tok)
                # Line 10
                jmax = S.argmax()
                Y, I_tok = YB[jmax], tokB[jmax]
                if debug:
                    mask_img = self.decode_masks((~masks1[jmax]).float())
                    masked_img = image_samples[-1]
                    masked_img = torch.clamp(masked_img * 0.7 + mask_img * 0.4, 0, 1)
                    image_samples.append(masked_img)
                    image_samples.append(self.decode_images(I_tok))

                if dynamic:
                    if S[jmax] > Smax:
                        tmax = t
                        Smax = S[jmax]
                        Imax = I_tok
                    if t - tmax >= 5:  # dynamic termination
                        break
                else:
                    Imax = I_tok

            sample_toks.append(Imax)

        sample_toks = torch.cat(sample_toks, 0)
        return sample_toks, image_samples

    def get_image_tokens(self, image, reshape=True, insert_sep=False, which_vae='vae'):
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        if isinstance(image, list):
            assert len(image[0].shape) == 4, 'image should be list of 4d image tensors'
            image = torch.stack(image, dim=1)
        if len(image.shape) == 4:
            image = image.unsqueeze(1)
        is_raw_image = len(image.shape) == 5  # [B, T, C, H, W]
        if is_raw_image:
            b, t, c, h, w = image.shape
            image_size = vae.image_size
            assert (c, h, w) == (3, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'
            image = rearrange(image, 'b t c h w -> (b t) c h w')
            image = vae.get_codebook_indices(image)  # ((b t) n)
            if reshape:
                if insert_sep:
                    sep_index = self.image_token_lut['[SEP]']
                    image = rearrange(image, '(b t) n -> b t n', t=t)
                    image = torch.cat((image, torch.empty((b, t, 1), device=image.device).long().fill_(sep_index)),
                                      dim=2)  # (B, T, N+1)
                    image = rearrange(image, 'b t n -> b (t n)')
                else:
                    image = rearrange(image, '(b t) n -> b (t n)', t=t)
        return image

    @torch.no_grad()
    def recon_images(self, images, which_vae='vae'):
        if which_vae == 'cvae' and self.cvae is not None:
            vae = self.cvae
        else:
            vae = self.vae
        img_seq = self.get_image_tokens(images, reshape=False, which_vae=which_vae)  # (B*T, N)
        images = vae.decode(img_seq)
        return images

    @torch.no_grad()
    def get_codebook_emb(self, images, which_vae='vae'):
        b, t, c, h, w = images.shape
        img_seq = self.get_image_tokens(images, reshape=False, which_vae=which_vae)
        img_code = rearrange(img_seq, '(b t) n -> b t n', t=t)
        img_embd = self.image_emb(img_code)
        return img_code, img_embd

    def random_erase_codebook(self, image, eraser, erase_half=False):
        image = rearrange(image, 'b (t h w) -> b t h w', h=self.image_fmap_size, w=self.image_fmap_size)
        if erase_half:
            image_ = image
            image_[:, :, self.image_fmap_size // 2:, :] = self.image_token_lut['[MASK]']  # erase the lower half
        else:
            image_ = torch.stack([eraser(c) for c in image], dim=0)
        image = rearrange(image_, 'b t h w -> b (t h w)', h=self.image_fmap_size, w=self.image_fmap_size)
        return image

    def erase_codebook_face(self, image, vc_mode, face_mode=None):
        image = rearrange(image, 'b (t h w) -> b t h w', h=self.image_fmap_size, w=self.image_fmap_size)
        if vc_mode == 'face_8x8':
            image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(image).long()
            if face_mode is None:
                face_mode = 'eyes_nose' if random.random() < 0.5 else 'mouth'
            if face_mode == 'eyes_nose':  # eyes and nose
                image_[:, :, 2:5, 1:7] = image[:, :, 2:5, 1:7]
            else:  # mouth
                image_[:, :, 5:7, 2:6] = image[:, :, 5:7, 2:6]
            image = image_
        elif vc_mode == 'face2_8x8':  # appearance and motion
            image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(
                image).long()
            image_[:, 0, ...] = image[:, 0, ...]
            image_[:, 1:, 2:6, 2:6] = image[:, 1:, 2:6, 2:6]
            image = image_
        elif vc_mode == 'face3_8x8':
            image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(image).long()
            image_[:, 0, ...] = image[:, 0, ...]
            image_[:, :, 2:6, 2:6] = image[:, :, 2:6, 2:6]
            image = image_
        elif vc_mode == 'mask_8x8' or vc_mode == 'mask2_8x8':
            # randomly pick a masking strategy
            if face_mode is None:
                which_strategy = np.random.choice([1, 2, 3], p=[0.5, 0.25, 0.25])
            else:
                which_strategy = 3
            image_ = None
            if which_strategy == 1:
                image_ = image  # no change
            elif which_strategy == 2:
                image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(image).long()
                image_[:, :, 2:6, 2:6] = image[:, :, 2:6, 2:6]
            elif which_strategy == 3:
                image_ = self.image_token_lut['[MASK]'] + torch.zeros_like(image).long()
                image_[:, :, 1:7, 1:7] = image[:, :, 1:7, 1:7]
            image = image_
        elif vc_mode == 'shape_4x4':
            image[:, :, 1:3, 1:3] = self.image_token_lut['[MASK]']
        else:
            raise NotImplementedError
        image = rearrange(image, 'b t h w -> b (t h w)', h=self.image_fmap_size, w=self.image_fmap_size)
        return image

    def get_special_token(self, tok_list, batch_size=1, device='cuda'):
        tok = torch.tensor(tok_list, dtype=torch.long, device=device)
        return tok.repeat(batch_size, 1)  # [B, len(tok_list)]

    def swap_one_frame_along_batch(self, tokens, t=1):
        tokens_shuffled = tokens.detach().clone()  # [b, n, c]
        b, n, c = tokens.shape
        tokens_shuffled = tokens_shuffled.reshape(b, t, n // t, -1)  # [b, t, n//t, c]
        idx = np.random.randint(0, t, b)  # for each batch, we randomly pick a frame index between [0, t-1] -> [b, 1]
        frames_shuffled = torch.cat(torch.chunk(tokens_shuffled[range(b), idx, ...], 2, dim=0)[::-1], dim=0)  # [b, n//t, c]
        tokens_shuffled[range(b), idx, ...] = frames_shuffled
        tokens_shuffled = tokens_shuffled.reshape(b, n, c)
        return tokens_shuffled

    # ======================= forward ========================== #
    def forward(
            self,
            text,
            visual=None,
            target=None,
            mask=None,
            return_loss=False,
            rel=False,
            vid=False,
            erase_visual=False,
            erase_visual_half=False,
            msm_strategy_prob=[0.7, 0.1, 0.1, 0.1],
            msm_bernoulli_prob=[0.2, 0.5],
            rel_no_fully_masked=False,
            vid_strategy_prob=[0.25, 0.25, 0.25, 0.25],
            negvc=False,
            visual_neg=None,
            text_neg=None,
            pc_prob=0,
            vc_mode=None,
            face_mode=None,
            visual_aug_mode=None,
            **kwargs,
    ):
        text_list = None
        if isinstance(text, list):
            text_list = text
            text = torch.stack(text, dim=1)  # [b, 2, text_seq_len]

        # visual and target are lists or 5d tensors (B, T, C, H, W)
        device = text[0].device
        batch_size = text.size()[0]
        if self.fixed_language_model is None:
            text_shape = text.shape
            if self.eval_mode == 'test_inter':
                text_shape = [text.shape[0], text.shape[-1]]
        else:  # NOTE: use embedding which takes a single token (from say RoBERTa)
            text_shape = [text.shape[0], 1]

        # NOTE: Prepend [REL]
        # self.before_control_tok = [0] -> REL
        before_tok = self.get_special_token(self.before_control_tok, batch_size, device)  # [B, 1] tensor
        before_emb = self.special_emb(before_tok)  # [B, 1, dim]
        before_emb += self.special_pos_emb(before_tok)  # [B, 1, dim]
        control_emb = before_emb
        control_seq_len = before_emb.shape[1]
        if negvc:
            control_neg_emb = before_emb

        # NOTE: make sure padding in text tokens get unique padding token id
        if self.fixed_language_model is None:
            assert text.shape[-1] == self.text_seq_len, \
                f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'

            # @Crane: encoding the texts
            if self.eval_mode == 'test_inter':
                text_range = torch.arange(self.text_seq_len, device=device) + (self.num_text_tokens - self.text_seq_len)
                text1 = text[:, 0]
                text2 = text[:, 1]
                text1 = torch.where(text1 == 0, text_range, text1)
                text1_emb = self.text_emb(text1)
                text2 = torch.where(text2 == 0, text_range, text2)
                text2_emb = self.text_emb(text2)

                alphas = [float(i) / (self.text_seq_len - 1) for i in range(self.text_seq_len)]
                text_emb = [(1 - alpha) * text1_emb[:, i:i+1] + alpha * text2_emb[:, i:i+1]
                            for i, alpha in enumerate(alphas)]
                text_emb = torch.cat(text_emb, dim=1)  # [b, text_seq_len, dim]
                text_emb += self.text_pos_emb(torch.arange(text_shape[1], device=device))
            else:
                text_range = torch.arange(self.text_seq_len, device=device) + (self.num_text_tokens - self.text_seq_len)
                text = torch.where(text == 0, text_range, text)
                text_emb = self.text_emb(text)  # [b, text_seq_len, dim]
                text_emb += self.text_pos_emb(torch.arange(text_shape[1], device=device))

        else:
            text_emb = self.text_feature_mapping(text)
            text_emb = text_emb.unsqueeze(1)

        control_emb = torch.cat((control_emb, text_emb), dim=1)  # [B, 1 + text_seq_len, dim]
        control_seq_len += text_emb.shape[1]  # 1 + text_seq_len
        if negvc:
            # NOTE: current text_neg does not guarantee to be neg
            text_neg = torch.where(text_neg == 0, text_range, text_neg)
            text_neg_emb = self.text_emb(text_neg)
            text_neg_emb += self.text_pos_emb(torch.arange(text_shape[1], device=device))
            control_neg_emb = torch.cat((control_neg_emb, text_neg_emb), dim=1)

        if self.num_visuals > 0:
            if exists(visual) and len(visual):  # visual: [B, T, C, H, W]
                # negative augmentation: color jittering
                if visual_aug_mode == 'motion_color' and random.random() < 0.9:
                    visual_ = visual.detach().clone()
                    visual_[:, 1:, ...] = warp_video_with_color(visual[:, 1:, ...])
                    visual = visual_
                # get the image tokens: [b, t * (n+1)] if insert_sep, else [b, t * n]
                visual = self.get_image_tokens(visual, insert_sep=self.insert_sep, which_vae='cvae')
                if erase_visual:
                    visual = self.random_erase_codebook(visual, self.visual_eraser, erase_visual_half)
                if vc_mode is not None:
                    visual = self.erase_codebook_face(visual, vc_mode, face_mode)
            else:
                visual = torch.empty(batch_size, self.visual_seq_len, device=device).long().fill_(
                    self.image_token_lut['[MASK]'])
            visual_emb = self.visual_emb(visual) if self.visual_emb else self.image_emb(visual)  # [b, num_image_token + 2, dim]

            visual_pos_emb = self.visual_pos_emb(visual_emb)
            visual_emb += visual_pos_emb
            control_emb = torch.cat((control_emb, visual_emb), dim=1)
            control_seq_len += visual.shape[1]

        # NOTE: Append [ST1, VID]
        after_tok = self.get_special_token(self.after_control_tok, batch_size, device)  # [b, 2, dim]
        after_emb = self.special_emb(after_tok)
        after_emb += self.special_pos_emb(after_tok)
        control_emb = torch.cat((control_emb, after_emb), dim=1)
        control_seq_len += after_emb.shape[1]
        if negvc:
            control_neg_emb = torch.cat((control_neg_emb, after_emb), dim=1)

        if not return_loss:
            return control_emb

        target_orig = None
        if exists(target) and len(target):
            target_orig = target.detach().clone()
            target = self.get_image_tokens(target)  # (B, T*N)

        # NOTE: Masked Sequence Modeling
        #   Masking strategies:
        #     (1) randomly mask a number of tokens according to Bernoulli distribution;
        #     (2) mask all tokens;
        #     (3) mask within boxed areas;
        #     (4) mask outside boxed areas;
        mask1_ = []
        not_fully_masked = torch.ones(batch_size, device=device)  # batch index for not fully masked
        for i in range(batch_size):
            # randomly choose a strategy
            which_strategy = np.random.choice([1, 2, 3, 4], p=msm_strategy_prob)
            if which_strategy == 1:
                p = np.random.uniform(*msm_bernoulli_prob)  # prob of keep GT tok
                mask1 = torch.bernoulli(torch.ones(self.target_seq_len, device=device) * p)  # keep GT if mask is 1
            elif which_strategy == 2:
                not_fully_masked[i] = 0
                mask1 = torch.zeros(self.target_seq_len, device=device)
            elif which_strategy == 3:
                mask1 = self.random_erasing(
                    torch.ones((self.num_targets, 1, self.image_fmap_size, self.image_fmap_size), device=device)
                ).reshape(-1)
            elif which_strategy == 4:
                mask1 = self.random_erasing(
                    torch.ones((self.num_targets, 1, self.image_fmap_size, self.image_fmap_size), device=device)
                ).reshape(-1)
                mask1 = 1 - mask1
            else:
                raise NotImplementedError

            if pc_prob > 0 and random.random() < pc_prob:
                t_overlap = random.randint(1, self.num_targets // 2)
                for tt in random.sample(range(self.num_targets), t_overlap):
                    mask1[self.image_seq_len * tt:self.image_seq_len * (tt + 1)] = 1
            # mask1: [target_seq_len]
            mask1_.append(mask1)  # [target_seq_len] * b

        mask1 = torch.stack(mask1_, 0).bool()  # [b, target_seq_len]
        target_masked = torch.where(mask1, target, self.image_token_lut['[MASK]'])
        target_emb_masked = self.image_emb(target_masked)
        target_pos_emb = self.target_pos_emb(target_emb_masked)
        tokens_msm = torch.cat((control_emb, target_emb_masked + target_pos_emb), dim=1)
        out = self.transformer_forward(tokens_msm)
        out_msm = out[:, control_seq_len:, :]  # target sequence: [b, n, dim]

        logits_msm = self.to_logits(out_msm)  # [b, n, num_img_tokens]
        loss_msm = F.cross_entropy(logits_msm[~mask1], target[~mask1])  # prediction loss

        # NOTE: Relevance Estimation Task
        if rel:
            assert text_shape[0] >= 2 and text_shape[0] % 2 == 0  # for REL swapping
            if negvc:
                tokens_neg_rel = torch.cat((control_neg_emb, target_emb_masked + target_pos_emb), dim=1)
                out_neg_rel = self.transformer_forward(tokens_neg_rel)
            else:
                control_emb_swap = swap(control_emb, 0)
                tokens_neg_rel = torch.cat([control_emb_swap, target_emb_masked + target_pos_emb], dim=1)
                out_neg_rel = self.transformer_forward(tokens_neg_rel)
                logits_pos_rel = self.to_logits_rel(out[:, self.rel_tok_index, :]).squeeze()  # [b, dim]
                logits_neg_rel = self.to_logits_rel(out_neg_rel[:, self.rel_tok_index, :]).squeeze()

            weight_pos = 1
            if rel_no_fully_masked:
                loss_rel_pos = F.binary_cross_entropy_with_logits(
                    logits_pos_rel,
                    torch.ones(batch_size, device=device),
                    reduction='none') * weight_pos
                loss_rel_neg = F.binary_cross_entropy_with_logits(
                    logits_neg_rel,
                    torch.zeros(batch_size, device=device),
                    reduction='none')
                num_valid_samples = max(1., not_fully_masked.sum())
                loss_rel = ((loss_rel_pos + loss_rel_neg) * not_fully_masked).sum() / num_valid_samples
            else:
                loss_rel = (F.binary_cross_entropy_with_logits(
                    logits_pos_rel, torch.ones(batch_size, device=device)) *
                            weight_pos + F.binary_cross_entropy_with_logits(
                            logits_neg_rel,
                            torch.zeros(batch_size, device=device)))
        else:
            loss_rel = torch.tensor(0.0, device=device)

        # NOTE: Continuity Estimation Task
        if vid and self.num_targets > 1:
            weight_pos = 1
            weight_neg = 1
            # get warped frames
            target_warp = warp(target_orig, vid_strategy_prob)
            target_warp = self.get_image_tokens(target_warp)
            target_warp_masked = torch.where(mask1, target_warp, self.image_token_lut['[MASK]'])
            target_emb_warp_masked = self.image_emb(target_warp_masked)
            tokens_neg_vid = torch.cat((control_emb, target_emb_warp_masked + target_pos_emb), dim=1)
            out_neg_vid = self.transformer_forward(tokens_neg_vid)
            out_pos_vid = out
            logits_pos_vid = self.to_logits_vid(out_pos_vid[:, self.vid_tok_index, :])  # [B, 1]
            logits_neg_vid = self.to_logits_vid(out_neg_vid[:, self.vid_tok_index, :])

            if rel_no_fully_masked:
                num_valid_samples = max(1., not_fully_masked.sum())
                loss_pos_vid = F.binary_cross_entropy_with_logits(logits_pos_vid,
                                                                  torch.ones(batch_size, 1, device=device),
                                                                  reduction='none').sum()
                loss_neg_vid = F.binary_cross_entropy_with_logits(logits_neg_vid,
                                                                  torch.zeros(batch_size, 1, device=device),
                                                                  reduction='none').sum()
                loss_vid = (weight_pos * loss_pos_vid + weight_neg * loss_neg_vid) / num_valid_samples

            else:
                loss_vid = F.binary_cross_entropy_with_logits(logits_pos_vid,
                                                              torch.ones(batch_size, 1, device=device)) * weight_pos + \
                           F.binary_cross_entropy_with_logits(logits_neg_vid,
                                                              torch.zeros(batch_size, 1, device=device)) * weight_neg
        else:
            loss_vid = torch.tensor(0.0, device=device)

        return loss_msm, loss_rel, loss_vid
