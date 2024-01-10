import logging
from contextlib import contextmanager
from math import ceil
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import lightning.pytorch as L
import numpy as np
import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers.wandb import WandbLogger
from safetensors.torch import load_file as load_safetensors
from torch import Tensor

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.models.autoencoder import AutoencodingEngine
from neurosis.modules.diffusion import (
    BaseDiffusionSampler,
    Denoiser,
    StandardDiffusionLoss,
    UNetModel,
)
from neurosis.modules.diffusion.hooks import LossHook
from neurosis.modules.diffusion.wrappers import OpenAIWrapper
from neurosis.modules.ema import LitEma
from neurosis.modules.encoders import GeneralConditioner
from neurosis.modules.encoders.embedding import AbstractEmbModel
from neurosis.trainer.util import EMATracker
from neurosis.utils import disabled_train, log_txt_as_img, np_text_decode

logger = logging.getLogger(__name__)


class DiffusionEngine(L.LightningModule):
    def __init__(
        self,
        model: UNetModel,
        denoiser: Denoiser,
        first_stage_model: AutoencodingEngine,
        conditioner: GeneralConditioner,
        sampler: Optional[BaseDiffusionSampler],
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
        loss_fn: Optional[StandardDiffusionLoss],
        ckpt_path: Optional[PathLike] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast: bool = False,
        input_key: str = "jpg",
        log_keys: Union[list, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        vae_batch_size: Optional[int] = None,
        forward_hooks: list[LossHook] = [],
    ):
        super().__init__()
        logger.info("Initializing DiffusionEngine")

        self.log_keys = log_keys
        self.input_key = input_key

        self.model = OpenAIWrapper(model, compile_model=compile_model)
        self.denoiser = denoiser
        self.sampler = sampler
        self.conditioner = conditioner

        self.optimizer = optimizer
        self.scheduler = scheduler

        # do first stage model setup
        logger.info("Loading first stage (VAE) model...")
        self._init_first_stage(first_stage_model)

        self.loss_fn = loss_fn
        self.forward_hooks = forward_hooks

        self.use_ema = use_ema
        if self.use_ema:
            logger.info("Using EMA")
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        else:
            self.model_ema = None

        self.scale_factor: float = scale_factor
        self.first_stage_autocast: bool = not disable_first_stage_autocast
        self.no_cond_log: bool = no_cond_log
        self.vae_batch_size = vae_batch_size
        self.loss_ema = EMATracker(alpha=0.02)

        if ckpt_path is not None:
            self.init_from_ckpt(Path(ckpt_path))

        self.save_hyperparameters(
            ignore=[
                "model",
                "denoiser",
                "first_stage_model",
                "conditioner",
                "sampler",
                "loss_fn",
                "optimizer",
                "scheduler",
            ]
        )
        for pl_logger in self.loggers:
            if isinstance(pl_logger, WandbLogger):
                pl_logger.experiment.config.update(self.hparams)

    def init_from_ckpt(self, path: Path) -> None:
        if path.suffix == ".safetensors":
            sd = load_safetensors(path)
        elif path.suffix in CHECKPOINT_EXTNS:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        else:
            raise NotImplementedError(f"Unknown checkpoint extension {path.suffix}")

        missing, unexpected = self.load_state_dict(sd, strict=False)
        logger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            logger.warn(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, model: AutoencodingEngine):
        model.eval()
        model.freeze()
        model.train = disabled_train
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z: Tensor) -> Tensor:
        z = 1.0 / self.scale_factor * z
        n_samples = self.vae_batch_size or z.shape[0]

        n_rounds = ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast(self.device.type, enabled=self.first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x: Tensor) -> Tensor:
        n_samples = self.vae_batch_size or x.shape[0]
        n_rounds = ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast(self.device.type, enabled=self.first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch) -> tuple[Tensor, dict[str, Tensor]]:
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        return loss

    def shared_step(self, batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss = self(x, batch)
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        # run any pre-step hooks
        for hook in self.forward_hooks:
            hook.pre_hook(self.trainer, self, batch, batch_idx)

        # run the actual step
        loss = self.shared_step(batch)

        # run any post-step hooks
        loss_dict = {}
        for hook in self.forward_hooks:
            loss, loss_dict = hook(self, batch, loss, loss_dict)

        # log the adjusted loss

        self.loss_ema.update(loss.mean().item())
        loss_dict.update({"train/loss": loss.mean(), "train/loss_ema": self.loss_ema.value})

        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            batch_size=batch[self.input_key].shape[0],
        )

        return loss.mean()

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    def configure_optimizers(self):
        param_groups = []
        unet_params = {"name": "UNet", "params": list(self.model.parameters())}
        # add initial_lr if set on the unet
        if hasattr(self.model, "base_lr") and self.model.base_lr is not None:
            logger.info(f"Setting initial_lr for unet to {self.model.base_lr:.2e}")
            unet_params["initial_lr"] = self.model.base_lr
        param_groups.append(unet_params)

        embedder: AbstractEmbModel
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                embedder_name = getattr(embedder, "name", embedder.__class__.__name__)
                logger.info(f"Adding {embedder_name} to trainable parameter groups")

                embedder_params = {"name": embedder_name, "params": list(embedder.parameters())}
                # add initial_lr if set on the embedder
                if hasattr(embedder, "base_lr") and embedder.base_lr is not None:
                    logger.info(f"Setting initial_lr for {embedder_name} to {embedder.base_lr:.2e}")
                    embedder_params["initial_lr"] = embedder.base_lr

                param_groups.append(embedder_params)

        optimizer = self.optimizer(param_groups)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        return optimizer

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        uc: Union[dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, list] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        def denoiser(input, sigma, c):
            return self.denoiser(self.model, input, sigma, c, **kwargs)

        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: dict[str, Tensor], n: int) -> dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        log_dict = dict()
        if self.no_cond_log is True:
            return log_dict

        wh = batch[self.input_key].shape[2:]

        embedder: AbstractEmbModel
        for embedder in self.conditioner.embedders:
            if (self.log_keys is None) or (embedder.input_key in self.log_keys):
                x = batch[embedder.input_key][:n]
                if isinstance(x, Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img(wh, x, size=min(wh[0] // 4, 96))
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img(wh, x, size=min(wh[0] // 20, 24))
                    else:
                        raise NotImplementedError("Tensor conditioning with dim > 2 not implemented")

                elif isinstance(x, list):
                    if isinstance(x[0], np.bytes_):
                        x = np_text_decode(x)
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img(wh, x, size=min(wh[0] // 20, 24))
                    else:
                        raise NotImplementedError(f"Conditioning log for list[{type(x[0])}] not implemented")

                else:
                    raise NotImplementedError(f"Conditioning log for input of {type(x)} not implemented")
                log_dict[embedder.input_key] = xc

        return log_dict

    @torch.no_grad()
    def log_images(
        self,
        batch: dict,
        num_img: int = 8,
        sample: bool = True,
        ucg_keys: list[str] = None,
        **kwargs,
    ) -> dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys

        log_dict = dict()
        x: Tensor = self.get_input(batch)
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        num_img = min(x.shape[0], num_img)

        x = x.to(self.device)[:num_img]
        log_dict["inputs"] = x

        z: Tensor = self.encode_first_stage(x)
        log_dict["reconstructions"] = self.decode_first_stage(z)
        log_dict.update(self.log_conditionings(batch, num_img))

        for k in c:
            if isinstance(c[k], Tensor):
                c[k], uc[k] = map(lambda y: y[k][:num_img].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=num_img, **kwargs)
            samples = self.decode_first_stage(samples)
            log_dict["samples"] = samples
        return log_dict
