import torch
from dalle2_pytorch import OpenAIClipAdapter
from dalle2_pytorch.dalle2_pytorch import l2norm
from dalle2_pytorch.optimizer import get_optimizer
from torchmetrics.classification.accuracy import Accuracy

import wandb
from src.models.base_model import BaseModule
from src.viz.points import plot_histogram

train_acc = Accuracy()
# pylint: disable=abstract-method,too-many-ancestors,arguments-renamed,line-too-long,arguments-differ,unused-argument


class Img2TextModule(BaseModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.clip = OpenAIClipAdapter(cfg.clip_model)
        self.image_embed_scale = cfg.image_embed_dim**0.5
        self.meaningful_class = None
        self.class_emb = None

    def on_train_start(self):
        self.meaningful_class = self.trainer.datamodule.meaningful_class
        meaningful_token = self.trainer.datamodule.meaningful_token
        with torch.no_grad():
            self.class_emb, _ = self.clip.embed_text(meaningful_token.to(self.device))
        self.class_emb = l2norm(self.class_emb)

    def get_real_data(self, src_data):
        # src_data: (img_tensor, label)
        src_img = src_data[0]
        uniform_idx = torch.randint(self.class_emb.shape[0], (src_img.shape[0],))
        trg_txt_emb = self.class_emb[uniform_idx]
        with torch.no_grad():
            src_img_emb, _ = self.clip.embed_image(src_img)
        src_img_emb *= self.image_embed_scale
        trg_txt_emb *= self.image_embed_scale
        return src_img_emb, trg_txt_emb

        # src_data, trg_data = batch
        # # src_data: (img_tensor, label)
        # # trg_data: (img_tensor, label)
        # src_img, trg_label = src_data[0], trg_data[1]
        # with torch.no_grad():
        #     src_img_emb, _ = self.clip.embed_image(src_img)
        #     trg_class = self.meaningful_class[trg_label.detach().cpu()]
        #     if self.global_step==1:
        #         print(trg_class)
        #     trg_token = tokenize(trg_class)
        #     trg_txt_emb, _ = self.clip.embed_text(trg_token.to(self.device))
        # src_img_emb *= self.image_embed_scale
        # trg_txt_emb *= self.image_embed_scale
        # return src_img_emb, trg_txt_emb

    def validation_step(self, batch, batch_idx):
        data, target = batch
        with torch.no_grad():
            emb_image, _ = self.clip.embed_image(data)
        adusted_emb = self.map_t(emb_image)

        similarity = (100.0 * l2norm(adusted_emb) @ self.class_emb.T).softmax(dim=-1)
        _, pred = similarity.topk(1, dim=-1)
        pred = pred.squeeze(1).detach().cpu()
        post_hoc_acc = train_acc(pred, target.cpu())
        train_acc.reset()

        similarity = (100.0 * l2norm(emb_image) @ self.class_emb.T).softmax(dim=-1)
        _, zero_shot_pred = similarity.topk(1, dim=-1)
        zero_shot_pred = zero_shot_pred.squeeze(1).detach().cpu()
        zero_shot_acc = train_acc(zero_shot_pred, target.cpu())
        train_acc.reset()

        return pred, zero_shot_pred, post_hoc_acc, zero_shot_acc, data.shape[0]

    def validation_epoch_end(self, outputs):
        pred_list = []
        zero_shot_pred_list = []
        correct_test_count = 0
        correct_zero_shot_count = 0
        for pred, zero_shot_pred, post_hoc_acc, zero_shot_acc, batch_size in outputs:
            pred_list.extend(list(pred.numpy()))
            zero_shot_pred_list.extend(list(zero_shot_pred.numpy()))
            correct_test_count += batch_size * post_hoc_acc
            correct_zero_shot_count += batch_size * zero_shot_acc
        accuracy = correct_test_count / len(pred_list)
        zero_shot_accuracy = correct_zero_shot_count / len(zero_shot_pred_list)
        self.log_dict(
            {
                "test_accuracy/post-hoc": accuracy,
                "test_accuracy/zero-shot": zero_shot_accuracy,
            }
        )
        torch.save(
            {
                "pred": pred_list,
                "clip_pred": zero_shot_pred_list,
                "acc": accuracy,
                "clip_zero_shot": zero_shot_accuracy,
            },
            f"pred_acc_{self.current_epoch+1}.pt",
        )

        hist_path = f"hist_{self.current_epoch+1}.png"
        plot_histogram(pred_list, num_class=self.class_emb.shape[0], path=hist_path)
        wandb.log({"histogram": wandb.Image(hist_path, caption="pred")})

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

    # def pretrain_identity(self, data, map_opt):
    #     loss = F.mse_loss(self.map_t(data), data)
    #     map_opt.zero_grad()
    #     loss.backward()
    #     map_opt.step()
    #     self.log_dict({"pretrain_loss/id_loss": loss})
