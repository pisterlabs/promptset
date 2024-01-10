# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keras implementation of StableDiffusionXL."""

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import correlate1d

from .clip_tokenizer import SimpleTokenizer
from .diffusion_model import DiffusionXLModel
from .image_decoder import ImageDecoder
from .image_encoder import ImageEncoder
from .long_prompt_weighting import get_weighted_text_embeddings
from .scheduler import Scheduler
from .text_encoder_laion import TextEncoderLaion, TextEncoderLaionProj
from .text_encoder_openai import TextEncoderOpenAi

MAX_PROMPT_LENGTH = 77


class StableDiffusionXLBase:
    """Base class for stable diffusion xl model."""

    def __init__(self, img_height=1024, img_width=1024, jit_compile=False,
                 active_lcm=False):
        self.img_height = img_height
        self.img_width = img_width
        # lazy initialize the component models and the tokenizer
        self._image_encoder = None
        self._text_encoder_laion = None
        self._text_encoder_laion_proj = None
        self._text_encoder_openai = None
        self._diffusion_model = None
        self._image_decoder = None
        self._tokenizer = None
        self.jit_compile = jit_compile
        self.active_lcm = active_lcm
        self.scheduler = Scheduler(active_lcm=active_lcm)

    def text_to_image(
            self,
            prompt,
            negative_prompt=None,
            batch_size=1,
            num_steps=50,
            unconditional_guidance_scale=7.5,
            seed=None,
            original_size=None,
            crops_coords_top_left=(0, 0),
            target_size=None,
            guidance_rescale=0.7,
            callback=None):
        encoded_text, add_text_embeds = self.encode_text(prompt)

        return self.generate_image(
            encoded_text,
            add_text_embeds,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            num_steps=num_steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            seed=seed,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            guidance_rescale=guidance_rescale,
            callback=callback)

    def image_to_image(
            self,
            prompt,
            negative_prompt=None,
            batch_size=1,
            num_steps=50,
            unconditional_guidance_scale=7.5,
            seed=None,
            reference_image=None,
            reference_image_strength=0.8,
            original_size=None,
            crops_coords_top_left=(0, 0),
            target_size=None,
            guidance_rescale=0.7,
            callback=None):
        encoded_text, add_text_embeds = self.encode_text(prompt)
        return self.generate_image(
            encoded_text,
            add_text_embeds,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            num_steps=num_steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            seed=seed,
            reference_image=reference_image,
            reference_image_strength=reference_image_strength,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            guidance_rescale=guidance_rescale,
            callback=callback)

    def inpaint(
            self,
            prompt,
            negative_prompt=None,
            batch_size=1,
            num_steps=50,
            unconditional_guidance_scale=7.5,
            seed=None,
            reference_image=None,
            reference_image_strength=0.8,
            inpaint_mask=None,
            mask_blur_strength=None,
            original_size=None,
            crops_coords_top_left=(0, 0),
            target_size=None,
            guidance_rescale=0.7,
            callback=None):
        encoded_text, add_text_embeds = self.encode_text(prompt)

        return self.generate_image(
            encoded_text,
            add_text_embeds,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            num_steps=num_steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            seed=seed,
            reference_image=reference_image,
            reference_image_strength=reference_image_strength,
            inpaint_mask=inpaint_mask,
            mask_blur_strength=mask_blur_strength,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            guidance_rescale=guidance_rescale,
            callback=callback)

    def encode_text(self, prompt):
        """Encodes a prompt into a latent text encoding.

        The encoding produced by this method should be used as the
        `encoded_text` parameter of `StableDiffusion.generate_image`. Encoding
        text separately from generating an image can be used to arbitrarily
        modify the text encoding prior to image generation, e.g. for walking
        between two prompts.

        Args:
            prompt: a string to encode, must be 77 tokens or shorter.
        Example:

        ```python
        from keras_cv.models import StableDiffusion

        model = StableDiffusionXL(img_height=1024, img_width=1024, jit_compile=True)
        encoded_text  = model.encode_text("Tacos at dawn")
        img = model.generate_image(encoded_text)
        ```
        """
        # Tokenize prompt (i.e. starting context)
        context_openai, _ = get_weighted_text_embeddings(self.tokenizer, self.text_encoder_openai, prompt,
                                                         model_max_length=MAX_PROMPT_LENGTH,
                                                         pad_token_id=49407)
        context_laion, add_text_embeds = get_weighted_text_embeddings(self.tokenizer, self.text_encoder_laion, prompt,
                                                                      model_max_length=MAX_PROMPT_LENGTH,
                                                                      pad_token_id=0,
                                                                      text_encoder_pool=self.text_encoder_laion_proj)
        return np.concatenate([context_openai, context_laion], axis=-1), add_text_embeds

    def gaussian_blur(self, image, radius=3, h_axis=1, v_axis=2):
        def build_filter1d(kernel_size):
            if kernel_size == 1:
                filter1d = [1]
            else:
                triangle = [[1, 1]]
                for i in range(1, kernel_size - 1):
                    cur_row = [1]
                    prev_row = triangle[i - 1]
                    for j in range(len(prev_row) - 1):
                        cur_row.append(prev_row[j] + prev_row[j + 1])
                    cur_row.append(1)
                    triangle.append(cur_row)
                filter1d = triangle[-1]
            filter1d = np.reshape(filter1d, (kernel_size,))
            return filter1d / np.sum(filter1d)

        weights = build_filter1d(radius)
        # Apply filter horizontally
        blurred_image = correlate1d(image, weights, axis=h_axis, output=None, mode="reflect", cval=0.0, origin=0)
        # Apply filter vertically
        blurred_image = correlate1d(blurred_image, weights, axis=v_axis, output=None, mode="reflect", cval=0.0,
                                    origin=0)
        return blurred_image

    @staticmethod
    def resize(image_array, new_h=None, new_w=None):
        h, w, c = image_array.shape
        if new_h == h and new_w == w:
            return image_array
        h_bounds = 0, h - 1
        w_bounds = 0, w - 1
        y = np.expand_dims(np.linspace(h_bounds[0], h_bounds[1], new_h), axis=-1)
        x = np.expand_dims(np.linspace(w_bounds[0], w_bounds[1], new_w), axis=0)
        # Calculate the floor and ceiling values of x and y
        x_floor = np.floor(x).astype(int)
        x_ceil = np.ceil(x).astype(int)
        y_floor = np.floor(y).astype(int)
        y_ceil = np.ceil(y).astype(int)
        # Clip the values to stay within the image bounds
        x_floor = np.clip(x_floor, w_bounds[0], w_bounds[1])
        x_ceil = np.clip(x_ceil, w_bounds[0], w_bounds[1])
        y_floor = np.clip(y_floor, h_bounds[0], h_bounds[1])
        y_ceil = np.clip(y_ceil, h_bounds[0], h_bounds[1])
        # Calculate the fractional part of x and y
        dx = x - x_floor
        dy = y - y_floor
        # Get the values of the four neighboring pixels
        dx = np.expand_dims(dx, axis=-1)
        dy = np.expand_dims(dy, axis=-1)
        q11 = image_array[y_floor, x_floor, :]
        q21 = image_array[y_floor, x_ceil, :]
        q12 = image_array[y_ceil, x_floor, :]
        q22 = image_array[y_ceil, x_ceil, :]
        # Perform bilinear interpolation
        top_interp = q11 * (1.0 - dx) + q21 * dx
        bottom_interp = q12 * (1.0 - dx) + q22 * dx
        interpolated = top_interp * (1.0 - dy) + bottom_interp * dy
        return interpolated

    def preprocessed_image(self, x):
        if type(x) is str:
            x = np.array(Image.open(x).convert("RGB"))
        else:
            x = np.asarray(x)
        image_array = self.resize(x, self.img_height, self.img_width)
        image_array = np.array(image_array, dtype=np.float32) / 255.0
        input_image_array = image_array[None, ..., :3]
        input_image_tensor = input_image_array * 2.0 - 1.0
        return input_image_array, input_image_tensor

    def preprocessed_mask(self, x, blur_radius=5):
        if type(x) is str:
            x = np.array(Image.open(x).convert("L"))
        else:
            x = np.asarray(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=-1)
        mask_array = self.resize(x, self.img_height, self.img_width)
        if mask_array.shape[-1] != 1:
            mask_array = np.mean(mask_array, axis=-1, keepdims=True)
        input_mask_array = np.array(mask_array, dtype=np.float32) / 255.0
        if blur_radius is not None:
            input_mask_array = self.gaussian_blur(input_mask_array, radius=blur_radius, h_axis=0, v_axis=1)
        latent_mask_tensor = self.resize(input_mask_array, self.img_width // 8, self.img_height // 8)
        return np.expand_dims(input_mask_array, axis=0), np.expand_dims(latent_mask_tensor, axis=0)

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0, epsilon=1e-05):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/abs/2305.08891). See Section 3.4
        """
        std_text = np.std(noise_pred_text, axis=tuple(range(1, len(noise_pred_text.shape))), keepdims=True)
        std_cfg = np.std(noise_cfg, axis=tuple(range(1, len(noise_cfg.shape))), keepdims=True) + epsilon
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg
        return noise_cfg

    def generate_image(
            self,
            encoded_text,
            add_text_embeds,
            negative_prompt=None,
            batch_size=1,
            num_steps=50,
            unconditional_guidance_scale=7.5,
            diffusion_noise=None,
            seed=None,
            inpaint_mask=None,
            mask_blur_strength=None,
            reference_image=None,
            reference_image_strength=0.8,
            callback=None,
            original_size=None,
            crops_coords_top_left=(0, 0),
            guidance_rescale=0.0,
            target_size=None):
        """Generates an image based on encoded text.

        The encoding passed to this method should be derived from
        `StableDiffusion.encode_text`.

        Args:
            encoded_text: Tensor of shape (`batch_size`, 77, 768), or a Tensor
                of shape (77, 768). When the batch axis is omitted, the same
                encoded text will be used to produce every generated image.
            batch_size: int, number of images to generate, defaults to 1.
            negative_prompt: a string containing information to negatively guide
                the image generation (e.g. by removing or altering certain
                aspects of the generated image), defaults to None.
            num_steps: int, number of diffusion steps (controls image quality),
                defaults to 50.
            unconditional_guidance_scale: float, controlling how closely the
                image should adhere to the prompt. Larger values result in more
                closely adhering to the prompt, but will make the image noisier.
                Defaults to 7.5.
            diffusion_noise: Tensor of shape (`batch_size`, img_height // 8,
                img_width // 8, 4), or a Tensor of shape (img_height // 8,
                img_width // 8, 4). Optional custom noise to seed the diffusion
                process. When the batch axis is omitted, the same noise will be
                used to seed diffusion for every generated image.
            seed: integer which is used to seed the random generation of
                diffusion noise, only to be specified if `diffusion_noise` is
                None.

        Example:

        ```python
        from stable_diffusion_xl.stable_diffusion_xl import StableDiffusionXL

        batch_size = 8
        model = StableDiffusionXL(img_height=1024, img_width=1024, jit_compile=True)
        e_tacos = model.encode_text("Tacos at dawn")
        e_watermelons = model.encode_text("Watermelons at dusk")

        e_interpolated = tf.linspace(e_tacos, e_watermelons, batch_size)
        images = model.generate_image(e_interpolated, batch_size=batch_size)
        ```
        """
        if diffusion_noise is not None and seed is not None:
            raise ValueError(
                "`diffusion_noise` and `seed` should not both be passed to "
                "`generate_image`. `seed` is only used to generate diffusion "
                "noise when it's not already user-specified."
            )

        context = self._expand_tensor(encoded_text, batch_size)
        if negative_prompt is None:
            negative_prompt = ""
        unconditional_context, unconditional_add_text_embeds = self.encode_text(negative_prompt)
        unconditional_context = self._expand_tensor(unconditional_context, batch_size)

        if diffusion_noise is not None:
            diffusion_noise = np.squeeze(diffusion_noise)
            if len(diffusion_noise.shape) == 3:
                diffusion_noise = np.repeat(np.expand_dims(diffusion_noise, axis=0), batch_size, axis=0)
        # Iterative reverse diffusion stage
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps[::-1]
        init_time = None
        init_latent = None
        input_image_array = None
        input_mask_array = None
        latent_mask_tensor = None
        if inpaint_mask is not None:
            input_mask_array, latent_mask_tensor = self.preprocessed_mask(inpaint_mask, mask_blur_strength)
            if input_mask_array is None or latent_mask_tensor is None:
                print("wrong inpaint mask:{}".format(inpaint_mask))
        if reference_image is not None and (0. < reference_image_strength < 1.):
            input_image_array, input_image_tensor = self.preprocessed_image(reference_image)
            if input_image_tensor is not None:
                num_steps = int(num_steps * reference_image_strength + 0.5)
                init_time = timesteps[num_steps]
                init_latent = self.image_encoder.predict_on_batch(input_image_tensor)
                timesteps = timesteps[:num_steps]
            else:
                print("wrong reference image:{}".format(reference_image))
        latent = self._get_initial_diffusion_latent(batch_size=batch_size,
                                                    init_latent=init_latent,
                                                    init_time=init_time,
                                                    seed=seed,
                                                    noise=diffusion_noise)
        progbar = tf.keras.utils.Progbar(len(timesteps))
        iteration = 0
        if original_size is None:
            original_size = [self.img_height, self.img_width]
        if target_size is None:
            target_size = [self.img_height, self.img_width]
        add_time_ids = tf.expand_dims(
            tf.convert_to_tensor(list(list(original_size) + list(crops_coords_top_left) + list(target_size)),
                                 latent.dtype), axis=0)
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            time_emb = np.repeat(np.reshape(timestep, [1, -1]), batch_size, axis=0)
            if unconditional_guidance_scale > 0.0:
                unconditional_latent = self.diffusion_model.predict_on_batch(
                    [latent, time_emb, unconditional_context, add_time_ids, tf.zeros_like(add_text_embeds)])
                latent_text = self.diffusion_model.predict_on_batch(
                    [latent, time_emb, context, add_time_ids, add_text_embeds])
                latent = unconditional_latent + unconditional_guidance_scale * (
                        latent_text - unconditional_latent)
                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/abs/2305.08891
                    latent = self.rescale_noise_cfg(latent, latent_text, guidance_rescale=guidance_rescale)
            else:
                latent = self.diffusion_model.predict_on_batch(
                    [latent, time_emb, context, add_time_ids, add_text_embeds])
            latent = self.scheduler.step(latent, timestep, latent_prev)
            if latent_mask_tensor is not None and init_latent is not None:
                latent_orgin = self._get_initial_diffusion_latent(batch_size=batch_size,
                                                                  init_latent=init_latent,
                                                                  init_time=timestep,
                                                                  seed=seed,
                                                                  noise=diffusion_noise)
                latent = latent_orgin * (1. - latent_mask_tensor) + latent * latent_mask_tensor
            iteration += 1
            if callback is not None:
                callback(iteration)
            progbar.update(iteration)

        # Decoding stage
        decoded = self.image_decoder.predict_on_batch(latent)
        decoded = np.array(((decoded + 1.) * 0.5), dtype=np.float32)
        if input_mask_array is not None and input_image_array is not None:
            decoded = input_image_array * (1. - input_mask_array) + decoded * input_mask_array
        return np.clip(decoded * 255., 0, 255).astype("uint8")

    def _expand_tensor(self, text_embedding, batch_size):
        """Extends a tensor by repeating it to fit the shape of the given batch
        size."""
        text_embedding = np.squeeze(text_embedding)
        if len(text_embedding.shape) == 2:
            text_embedding = np.repeat(
                np.expand_dims(text_embedding, axis=0), batch_size, axis=0
            )
        return text_embedding

    @property
    def image_encoder(self):
        pass

    @property
    def text_encoder_openai(self):
        pass

    @property
    def text_encoder_laion(self):
        pass

    @property
    def text_encoder_laion_proj(self):
        pass

    @property
    def diffusion_model(self):
        pass

    @property
    def image_decoder(self):
        pass

    @property
    def tokenizer(self):
        """tokenizer returns the tokenizer used for text inputs.
        Can be overriden for tasks like textual inversion where the tokenizer
        needs to be modified.
        """
        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizer()
        return self._tokenizer

    def _get_initial_diffusion_noise(self, batch_size, seed):
        if seed is not None:
            try:
                seed = int(seed)
            except:
                seed = None
            return tf.random.stateless_normal(
                (batch_size, self.img_height // 8, self.img_width // 8, 4),
                seed=[seed, seed],
            )
        else:
            return tf.random.normal(
                (batch_size, self.img_height // 8, self.img_width // 8, 4)
            )

    def _get_initial_diffusion_latent(self, batch_size, init_latent=None, init_time=None, seed=None,
                                      noise=None):
        if noise is None:
            noise = self._get_initial_diffusion_noise(batch_size, seed=seed)
        if init_latent is None:
            latent = noise
        else:
            latent = self.scheduler.signal_rates[init_time] * np.repeat(init_latent, batch_size, axis=0) + \
                     self.scheduler.noise_rates[init_time] * noise
        return latent

    @staticmethod
    def _get_pos_ids():
        return np.asarray([list(range(MAX_PROMPT_LENGTH))], dtype=np.int32)


class StableDiffusionXL(StableDiffusionXLBase):
    """Keras implementation of Stable Diffusion.

    Note that the StableDiffusionXL API, as well as the APIs of the sub-components
    of StableDiffusionXL (e.g. ImageEncoder, DiffusionModel) should be considered
    unstable at this point. We do not guarantee backwards compatability for
    future changes to these APIs.

    Stable Diffusion is a powerful image generation model that can be used,
    among other things, to generate pictures according to a short text
    description (called a "prompt").

    Arguments:
        img_height: int, height of the images to generate, in pixel. Note that
            only multiples of 128 are supported; the value provided will be
            rounded to the nearest valid value. Defaults to 1024.
        img_width: int, width of the images to generate, in pixel. Note that
            only multiples of 128 are supported; the value provided will be
            rounded to the nearest valid value. Defaults to 1024.
        jit_compile: bool, whether to compile the underlying models to XLA.
            This can lead to a significant speedup on some systems. Defaults to
            False.

    Example:

    ```python
    from stable_diffusion_xl.stable_diffusion_xl import StableDiffusionXL
    from PIL import Image

    model = StableDiffusionXL(img_height=1024, img_width=1024, jit_compile=True)
    img = model.text_to_image(
        prompt="A beautiful horse running through a field",
        batch_size=1,  # How many images to generate at once
        num_steps=25,  # Number of iterations (controls image quality)
        seed=123,  # Set this to always get the same image from the same prompt
    )
    Image.fromarray(img[0]).save("horse.png")
    print("saved at horse.png")
    ```

    References:
    - [About Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement)
    - [Original implementation](https://github.com/CompVis/stable-diffusion)
    """  # noqa: E501

    def __init__(
            self,
            img_height=1024,
            img_width=1024,
            jit_compile=True,
            unet_ckpt=None,
            text_encoder_ckpt=None,
            text_encoder2_ckpt=None,
            vae_ckpt=None,
    ):
        super().__init__(img_height, img_width, jit_compile)
        self.unet_ckpt = unet_ckpt
        self.text_encoder_ckpt = text_encoder_ckpt
        self.text_encoder2_ckpt = text_encoder2_ckpt
        self.vae_ckpt = vae_ckpt

    @property
    def text_encoder_openai(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.
        """
        if self._text_encoder_openai is None:
            self._text_encoder_openai = TextEncoderOpenAi(MAX_PROMPT_LENGTH, ckpt_path=self.text_encoder_ckpt)
            if self.jit_compile:
                self._text_encoder_openai.compile(jit_compile=True)
        return self._text_encoder_openai

    @property
    def text_encoder_laion(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.
        """
        if self._text_encoder_laion is None:
            self._text_encoder_laion = TextEncoderLaion(MAX_PROMPT_LENGTH, ckpt_path=self.text_encoder2_ckpt)
            if self.jit_compile:
                self._text_encoder_laion.compile(jit_compile=True)
        return self._text_encoder_laion

    @property
    def text_encoder_laion_proj(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.
        """
        if self._text_encoder_laion_proj is None:
            self._text_encoder_laion_proj = TextEncoderLaionProj(ckpt_path=self.text_encoder2_ckpt)
            if self.jit_compile:
                self._text_encoder_laion_proj.compile(jit_compile=True)
        return self._text_encoder_laion_proj

    @property
    def diffusion_model(self):
        """diffusion_model returns the diffusion model with pretrained weights.
        Can be overriden for tasks where the diffusion model needs to be
        modified.
        """
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionXLModel(
                self.img_height, self.img_width, ckpt_path=self.unet_ckpt)
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self._diffusion_model

    @property
    def image_encoder(self):
        """image_encoder returns the VAE Encoder with pretrained weights."""
        if self._image_encoder is None:
            self._image_encoder = ImageEncoder(ckpt_path=self.vae_ckpt)
            if self.jit_compile:
                self._image_encoder.compile(jit_compile=True)
        return self._image_encoder

    @property
    def image_decoder(self):
        """decoder returns the diffusion image decoder model with pretrained
        weights. Can be overriden for tasks where the decoder needs to be
        modified.
        """
        if self._image_decoder is None:
            self._image_decoder = ImageDecoder(self.img_height, self.img_width, ckpt_path=self.vae_ckpt)
            if self.jit_compile:
                self._image_decoder.compile(jit_compile=True)
        return self._image_decoder
