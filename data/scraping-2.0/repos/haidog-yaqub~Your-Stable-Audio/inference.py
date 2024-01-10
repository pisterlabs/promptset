import os

import pandas as pd
import torch
import torchaudio
import numpy as np
import soundfile as sf
from tqdm import tqdm
from utils import scale_shift, scale_shift_re, align_seq


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@torch.no_grad()
def inference(autoencoder, unet, tokenizer, text_encoder, params, scheduler,
              text, audio_frames,
              guidance_scale=2, guidance_rescale=0.7,
              ddim_steps=50, eta=1, random_seed=2023,
              device='cuda',
              ):
    text_batch = tokenizer(text,
                           max_length=tokenizer.model_max_length,
                           padding=True, truncation=True, return_tensors="pt")
    text, text_mask = text_batch.input_ids.to(device), text_batch.attention_mask.to(device)
    text = text_encoder(input_ids=text, attention_mask=text_mask)[0]

    codec_dim = params.unet.codec_channels
    unet.eval()
    generator = torch.Generator(device=device).manual_seed(random_seed)
    scheduler.set_timesteps(ddim_steps)

    # init noise
    noise = torch.randn((1, codec_dim, audio_frames), generator=generator, device=device)
    latents = noise

    if guidance_scale is not None:
        uncond_text, uncond_mask = unet.get_cfg_emb(text, text_mask)
        text = torch.cat([text, uncond_text], dim=0)
        text_mask = torch.cat([text_mask, uncond_mask], dim=0)

    for t in scheduler.timesteps:
        latents = scheduler.scale_model_input(latents, t)
        latent_model_input = torch.cat([latents] * 2) if guidance_scale else latents
        output_pred = unet(latent_model_input, t, text, text_mask, train_cfg=False)

        if guidance_scale:
            output_text, output_uncond = output_pred.chunk(2)
            output_pred = output_uncond + guidance_scale * (output_text - output_uncond)
            if guidance_rescale > 0.0:
                output_pred = rescale_noise_cfg(output_pred, output_text,
                                                guidance_rescale=guidance_rescale)

        latents = scheduler.step(model_output=output_pred, timestep=t, sample=latents,
                                 eta=eta, generator=generator).prev_sample

    pred = scale_shift_re(latents, params.diff.scale, params.diff.shift)
    pred_wav = autoencoder.decoder(pred)
    return pred_wav


@torch.no_grad()
def eval_stable_audio(autoencoder, unet, tokenizer, text_encoder, params, scheduler,
                      val_df,
                      audio_frames,
                      guidance_scale=3, guidance_rescale=0.7,
                      ddim_steps=50, eta=1, random_seed=2023,
                      device='cuda',
                      epoch=0, save_path='logs/eval/', val_num=5):
    val_df = pd.read_csv(val_df)

    save_path = save_path + str(epoch) + '/'
    os.makedirs(save_path, exist_ok=True)

    for i in tqdm(range(len(val_df))):
        row = val_df.iloc[i]
        text = row['caption']

        pred = inference(autoencoder, unet, tokenizer, text_encoder, params, scheduler,
                         text, audio_frames,
                         guidance_scale, guidance_rescale,
                         ddim_steps, eta, random_seed,
                         device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)

        sf.write(save_path + text + '.wav', pred, samplerate=params.data.sr)

        if i + 1 >= val_num:
            break
