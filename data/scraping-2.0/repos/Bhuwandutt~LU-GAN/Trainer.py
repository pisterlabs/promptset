from typing import Optional
import os
# from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch

from OpenAIDataset import OpenAIDataset, ViewConsistencyDataset
from Decoder import Decoder, MultiscaleDecoder, PDecoder
from torch.utils.data import DataLoader
from Encoder import Encoder
from Siamese_Layer import *
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from Discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter
import datetime
import json
from tqdm import tqdm
from tensorboardcolab import TensorBoardColab

global tb = TensorBoardColab()
# class OPENIDataModule(pl.LightningDataModule):
#
# PyTorch lightning (https://www.pytorchlightning.ai) module for more compact and clean coding for Dataset setup
# and object creation.
#
#
#     def __init__(self, batch_size: int = 12, shuffle: bool = False):
#         super(OPENIDataModule, self).__init__()
#
#         self.siamese_set = None  # DataSet for training View Consistency Network
#         self.dataset = None  # DataSet containing all the images and reports
#         self.val_set = None  # Validation Dataset Split ( For changing the split ratio, see pre-processiong.py)
#         self.train_set = None  # Training Dataset
#         self.test_set = None  # Testing Dataset
#         self.batch_size = batch_size
#
#     def setup(self, stage: Optional[str] = 'train'):
#         if stage == 'train':
#             self.train_set = OpenAIDataset(file_name='indiana_reports_train',
#                                            transform=None)  # Transform is set to none but Transformation happens in
#             # DataClass
#         if stage == 'validation':
#             self.val_set = OpenAIDataset(file_name='indiana_reports_val',
#                                          transform=None)
#
#         if stage == 'test':
#             self.test_set = OpenAIDataset(file_name='indiana_reports_test',
#                                           transform=None)
#         if stage == 'sia':
#             self.siamese_set = ViewConsistencyDataset(file_name='indiana_reports_cleaned')
#         else:
#             self.dataset = OpenAIDataset(file_name='indiana_reports_cleaned',
#                                          transform=None)
#
#     def train_dataloader(self):
#         return DataLoader(self.train_set)
#
#     def test_dataloader(self):
#         return DataLoader(self.test_set)
#
#     def val_dataloader(self) -> EVAL_DATALOADERS:
#         return DataLoader(self.val_set)
#
#     def full_dataloader(self):
#         return DataLoader(self.dataset)
#

class Trainer:
    train_set: OpenAIDataset

    def __init__(self):
        super(Trainer, self).__init__()
        self.siamese_set = None  # DataSet for training View Consistency Network
        self.S_DataLoader = None  # DataSet containing all the images and reports
        self.val_set = None  # Dataset for Validation Set
        self.test_set = None
        self.DISP_FREQs = [10, 20, 30, 40]
        self.writer = SummaryWriter(os.path.join("runs"), 'Text-to-image XRayGAN OPENI256')
        # Dataloader setup (for implementation, see function setup()
        self.train_dataloader = self.setup('train')
        self.sia_dataloader = self.setup('sia')
        # Set up the DataModule for training by explicit
        self.S_optimizer = None
        self.S_lr_scheduler = None
        self.D_lr_scheduler = None
        self.D_optimizer = None
        self.G_lr_scheduler = None
        self.G_optimizer = None
        self.D_L = None
        self.D_F = None
        self.D_checkpoint = None
        self.decoder_checkpoint = os.getcwd()+'/checkpoint/'
        self.encoder_checkpoint = os.getcwd()+'/checkpoint/'
        self.embednet = None
        self.decoder_F = None
        self.decoder_L = None
        self.encoder = None  # Change to LinkBERT

        self.image_size = [256, 256]  # The resolution of generated image
        self.device = torch.device('cuda')
        self.G_LR = [0.0003, 0.0003, 0.0002, 0.0001]    # Generator learning rates
        self.D_LR = [0.0003, 0.0003, 0.0002, 0.0001]    # Decoder leaning rate
        self.LR_DECAY_EPOCH = [[45], [45, 70], [45, 70, 90], [45, 70, 90]]
        self.S_LR = 0.01  # Siamese Learning Rate, a.k.a. the Discriminator Layer
        self.MAX_EPOCH = [1, 1, 1, 1]
        self.SIAMESE_EPOCH = [8, 10, 10, 12]

        # Loss Function
        self.G_criterion = nn.MSELoss().to(self.device)  # Mean Square Error Loss
        self.S_criterion = nn.BCELoss().to(self.device)

        self.base_size = 32
        self.P_ratio = int(np.log2(self.image_size[0] // self.base_size))
        self.base_ratio = int(np.log2(self.base_size))
        print("Number of Decoders", self.P_ratio + 1)
        print("Number of Discriminator", self.P_ratio + 1)

        self.define_nets()
        # self.encoder = nn.DataParallel(self.encoder, device_ids=self.gpus)
        # self.decoder_L = nn.DataParallel(self.decoder_L, device_ids=self.gpus)
        # self.decoder_F = nn.DataParallel(self.decoder_F, device_ids=self.gpus)
        # self.embednet = nn.DataParallel(self.embednet, device_ids=self.gpus)
        # self.load_model()

    def setup(self, stage: Optional[str] = 'train'):

        if stage == 'train':
            self.train_set = OpenAIDataset(file_name='indiana_reports_train')
            return DataLoader(self.train_set)

        if stage == 'test':
            self.test_set = OpenAIDataset(file_name='indiana_reports_test')
            return DataLoader(self.test_set)

        if stage == 'val':
            self.val_set = OpenAIDataset(file_name='indiana_reports_val')
            return DataLoader(self.val_set)
        if stage == 'sia':
            self.siamese_set = ViewConsistencyDataset(file_name='indiana_reports_test')
            return DataLoader(self.siamese_set, drop_last=True)
            
        else:
            self.S_DataLoader = OpenAIDataset(file_name='indiana_reports_cleaned')

    def define_nets(self):
        # Comment the encoder out
        self.encoder = Encoder(feature_base_dim=512).to(self.device)

        decoders_F = []
        decoders_L = []
        first_decoder = Decoder(input_dim=512,
                                feature_base_dim=512, uprate=self.base_ratio).to(self.device)
        # first_decoder.apply(init_weights)
        decoders_F.append(first_decoder)
        decoders_L.append(first_decoder)

        for i in range(1, self.P_ratio + 1):
            nf = 128
            Pdecoder = PDecoder(input_dim=512,
                                feature_base_dim=nf).to(self.device)
            # decoder.apply(init_weights)
            decoders_F.append(Pdecoder)
            decoders_L.append(Pdecoder)

        self.decoder_L = MultiscaleDecoder(decoders_L).to(device=self.device)
        self.decoder_F = MultiscaleDecoder(decoders_F).to(device=self.device)
        self.embednet = Classifinet(backbone='alex').to(device=self.device)

    def define_opt(self, layer_id):

        # # Define optimizer
        # print(next(self.encoder.parameters()))
        # print(next(self.decoder_L.parameters()))
        # print(next(self.decoder_F.parameters()))

        self.G_optimizer = torch.optim.Adam([{'params': self.encoder.parameters()}] +
                                            [{'params': self.decoder_F.parameters()}],
                                            lr=self.G_LR[layer_id], betas=(0.9, 0.999))
        self.G_lr_scheduler = MultiStepLR(self.G_optimizer,
                                          milestones=self.LR_DECAY_EPOCH[layer_id],
                                          gamma=0.2)

        self.D_optimizer = self.G_optimizer
        self.D_lr_scheduler = MultiStepLR(self.D_optimizer,
                                          milestones=self.LR_DECAY_EPOCH[layer_id],
                                          gamma=0.2)
        self.S_optimizer = torch.optim.Adam(self.embednet.parameters(),
                                            lr=self.S_LR,
                                            betas=(0.9, 0.999))
        self.S_lr_scheduler = StepLR(self.S_optimizer, step_size=10, gamma=0.2)

    def check_create_checkpoint(self):
        # Check for the checkpoint path exists or not
        # If not exist, create folder
        if not os.path.exists(self.encoder_checkpoint):
            os.makedirs(self.encoder_checkpoint)
        if not os.path.exists(self.decoder_checkpoint):
            os.makedirs(self.decoder_checkpoint)
        if not os.path.exists(self.D_checkpoint):
            os.makedirs(self.D_checkpoint)

        def load_model(self):

            if os.path.exists(self.encoder_resume):
                print("load checkpoint {}".format(self.encoder_resume))
                self.encoder.load_state_dict(torch.load(self.encoder_resume))
            else:
                print("checkpoint do not exists {}".format(self.encoder_resume))

            if os.path.exists(self.decoder_resume_F):
                print("load checkpoint {}".format(self.decoder_resume_F))
                self.decoder_F.load_state_dict(torch.load(self.decoder_resume_F))
            else:
                print("checkpoint do not exists {}".format(self.decoder_resume_F))

            if os.path.exists(self.decoder_resume_L):
                print("load checkpoint {}".format(self.decoder_resume_L))
                self.decoder_L.load_state_dict(torch.load(self.decoder_resume_L))
            else:
                print("checkpoint do not exists {}".format(self.decoder_resume_L))

    def define_D(self, layer_id):

        # Initialize a series of Discriminator'''

        dr = self.base_ratio - 2 + layer_id
        self.D_F = Discriminator(base_feature=64,
                                 txt_input_dim=512,
                                 down_rate=dr).to(self.device)

        self.D_L = Discriminator(base_feature=64,
                                 txt_input_dim=512,
                                 down_rate=dr).to(self.device)

    @staticmethod
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def Loss_on_layer(self, image, finding_input_ids, impression_input_ids,finding_attention_mask,impression_attention_mask, layer_id, decoder):

        txt_emded = self.encoder(finding_input_ids, impression_input_ids,finding_attention_mask,impression_attention_mask)
        # print("Input Image", image.shape)
        r_image = F.interpolate(image, size=(2 ** layer_id) * 32)
        # print("r_image", r_image.size())
        self.G_optimizer.zero_grad()
        pre_image = decoder(txt_emded, layer_id)
        # pre_image = F.interpolate(pre_image, size=(2*layer_id)*self.base_size)
        # print("Pre Image", pre_image.size())
        loss = self.G_criterion(pre_image.float(), r_image.float())
        loss.backward()
        self.G_optimizer.step()
        return loss, pre_image, r_image

    def train_Siamese_layer(self, layer_id):
        # DISP_FREQ = self.DISP_FREQs[layer_id]
        for epoch in range(self.SIAMESE_EPOCH[layer_id]):
            self.embednet.train()
            print('VCN Epoch [{}/{}]'.format(epoch, self.SIAMESE_EPOCH[layer_id]))
            for idx, batch in enumerate(self.sia_dataloader):

                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)
                label = batch['label'].to(self.device)

                r_image_f = F.interpolate(image_f, size=(2 ** layer_id)*32)
                r_image_l = F.interpolate(image_l, size=(2 ** layer_id)*32)

                self.S_optimizer.zero_grad()
                pred = self.embednet(r_image_f, r_image_l)
                loss = self.S_criterion(pred, label)
                loss.backward()
                self.S_optimizer.step()

                # if ((idx + 1) % DISP_FREQ == 0) and idx != 0:
                #
                # self.writer.add_scalar('Train_Siamese {}_loss'.format(layer_id),
                #                        loss.item(),
                #                        epoch * len(self.S_DataLoader) + idx)
                #
                # self.writer.add_images("Train_front_{}_Original".format(layer_id),
                #                        (r_image_f + 1) / 2,
                #                        epoch * len(self.S_DataLoader) + idx)
                # self.writer.add_images("Train_lateral_{}_Original".format(layer_id),
                #                        (r_image_l + 1) / 2,
                #                        epoch * len(self.S_DataLoader) + idx)

            self.S_lr_scheduler.step(epoch)

            self.embednet.eval()
            total = 0
            correct = 0
            for idx, batch in enumerate(self.sia_dataloader):
                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)
                label = batch['label'].to(self.device)
                r_image_f = F.interpolate(image_f, size=(2 ** layer_id)*32)
                r_image_l = F.interpolate(image_l, size=(2 ** layer_id)*32)

                pred = self.embednet(r_image_f, r_image_l)
                pred[pred > 0.5] = 1
                pred[pred <= 0.5] = 0

                total += pred.shape[0]
                correct += torch.sum(pred == label).item()

            acc = correct / total
            # acc = self.evaluate_Siamese(layer_id)

            print(print("Accuracy {}".format(acc)))
            SummaryWriter().add_scalar('Acc_Siamese_Layer {}'.format(layer_id),
                                       acc,
                                       epoch)

    @staticmethod
    def get_time(self):
        # Get local time for checkpoint saving
        return (str(datetime.datetime.now())[:-10]).replace(' ', '-').replace(':', '-')

    def cal_gradient_penalty(self,netD, real_data, fake_data, txt_emded, type='mixed', constant=1.0,
                             lambda_gp=10.0):
        """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
        Arguments:
            netD (network)              -- discriminator network
            real_data (tensor array)    -- real images
            fake_data (tensor array)    -- generated images from the generator
            device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0]))
                                            if self.gpu_ids else torch.device('cpu')
            type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
            constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
            lambda_gp (float)           -- weight for this loss
        Returns the gradient penalty loss
        """
        if lambda_gp > 0.0:
            if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
                interpolatesv = real_data
            elif type == 'fake':
                interpolatesv = fake_data
            elif type == 'mixed':
                alpha = torch.rand(real_data.shape[0], 1, device='cuda')
                alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                    *real_data.shape)
                interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
                interpolatesv.requires_grad_(True)

            else:
                raise NotImplementedError('{} not implemented'.format(type))

            disc_interpolates = netD(interpolatesv, txt_emded)

            # grad() Computes and returns the sum of gradients of outputs with respect to the inputs.
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                            grad_outputs=torch.ones(disc_interpolates.size(),
                                                                    dtype=torch.float).to(device=self.device),
                                            create_graph=True, retain_graph=True, only_inputs=True)

            # print("Gradient 1", (gradients))
            gradients = gradients[0].view(real_data.size(0), -1)  # flatten the data
            gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
            #print("Gradient 1", gradients.size())
            # added eps
            return gradient_penalty, gradients[0]
        else:
            return 0.0, None

    def Loss_on_layer_GAN(self, image, finding_input_ids, impression_input_ids, finding_attention_mask,impression_attention_mask, layer_id, decoder, D):
        # '''
        # Pretrain genertaor with batch
        # :image image batch
        # :text text batch
        # '''

        global D_loss, G_loss
        image = F.interpolate(image, size=(2 ** layer_id) * self.base_size)
        txt_emded = self.encoder(finding_input_ids,impression_input_ids,finding_attention_mask, impression_attention_mask)  # Change the Encoder as LinkBERT

        pre_image = decoder(txt_emded, layer_id)

        # Train Discriminator
        for _ in range(1):
            
            gradient_penalty, gradients = self.cal_gradient_penalty(netD=D,
                                                                    real_data=image,
                                                                    fake_data=pre_image,
                                                                    txt_emded=txt_emded
                                                                    )
            self.D_optimizer.zero_grad()
            self.G_optimizer.zero_grad()

            pre_fake = D(pre_image, txt_emded)
            pre_real = D(image, txt_emded)
            
            # netD, real_data, fake_data, txt_emded, type='mixed', constant=1.0,
            #                              lambda_gp=10.0):
            adv_loss = -1 * pre_fake.mean()
            D_loss = pre_fake.mean() - pre_real.mean() + gradient_penalty
            content_loss = 100 * self.G_criterion(pre_image.float(),
                                                  image.float())
            # D_loss=D_loss.to('mps')
            torch.autograd.set_detect_anomaly(True)
            # print(f'D_Loss:{D_loss.size()}')

            D_loss.backward(retain_graph=True)
            adv_loss.backward(retain_graph=True)
            content_loss.backward(retain_graph=True)
            G_loss = content_loss + adv_loss

            self.D_optimizer.step()
            self.G_optimizer.step()
        return D_loss, G_loss, pre_image, image

    def train_layer(self, layer_id):
        DISP_FREQ = self.DISP_FREQs[layer_id]
        for epoch in range(1):
            print('Generator Epoch [{}/1]'.format(epoch+1))
            self.encoder.train()  # Train the encoder layer , comment this out
            self.decoder_F.train()
            self.decoder_L.train()
            for idx, batch in tqdm(enumerate(self.train_dataloader)):
                # print("Train")

                finding_input_ids = batch['finding_input_ids'].to(self.device)
                impression_input_ids = batch['impression_input_ids'].to(self.device)
                finding_attention_mask = batch['finding_attn'].to(self.device)
                impression_attention_mask = batch['impression_attn'].to(self.device)
                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)
                finding = {'input_ids':finding_input_ids, 'attention_mask': finding_attention_mask}
                impression = {'input_ids':impression_input_ids, 'attention_mask': impression_attention_mask}
                #
                # print(f"Finding Shape = {finding.size()}")
                # print(f'Impression shape = {impression.size()}')
                # print(f'Image Frontal shape = {image_f.size()}')
                # print(f'Image Lateral shape ={ image_l.size()}')

                loss_f,  pre_image_f, r_image_f = self.Loss_on_layer(image_f, finding_input_ids, impression_input_ids,finding_attention_mask,impression_attention_mask, layer_id,
                                                                    self.decoder_F)
                loss_l, pre_image_l, r_image_l = self.Loss_on_layer(image_l, finding_input_ids, impression_input_ids,finding_attention_mask,impression_attention_mask, layer_id,
                                                                    self.decoder_L)

                if ((idx + 1) % DISP_FREQ == 0) and idx != 0:
                    self.writer.add_scalar('Train_front {}_loss'.format(layer_id),
                                           loss_f.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('Train_lateral {}_loss'.format(layer_id),
                                           loss_l.item(),
                                           epoch * len(self.train_dataloader) + idx)

                    # write to tensorboard
                    self.writer.add_images("Train_front_{}_Original".format(layer_id),
                                           (r_image_f + 1) / 2,
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("Train_front_{}_Predicted".format(layer_id),
                                           (pre_image_f + 1) / 2,
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("Train_lateral_{}_Original".format(layer_id),
                                           (r_image_l + 1) / 2,
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("Train_lateral_{}_Predicted".format(layer_id),
                                           (pre_image_l + 1) / 2,
                                           epoch * len(self.train_dataloader) + idx)

            self.G_lr_scheduler.step(epoch)

    def train_GAN_layer(self, layer_id):
        DISP_FREQ = self.DISP_FREQs[layer_id]
        self.encoder.train() # Comment the file
        self.decoder_F.train()
        self.decoder_L.train()
        self.D_F.train()
        self.D_L.train()
        for epoch in range(self.MAX_EPOCH[layer_id]):
            print('GAN Epoch [{}/{}]'.format(epoch, self.MAX_EPOCH[layer_id]))
            for idx, batch in enumerate(self.train_dataloader):
                finding_input_ids = batch['finding_input_ids'].to(self.device)
                impression_input_ids = batch['impression_input_ids'].to(self.device)
                finding_attention_mask = batch['finding_attn'].to(self.device)
                impression_attention_mask = batch['impression_attn'].to(self.device)
                image_f = batch['image_F'].to(self.device)
                image_l = batch['image_L'].to(self.device)

                D_loss_f, G_loss_f, pre_image_f, image_f = self.Loss_on_layer_GAN(image_f, finding_input_ids, impression_input_ids,finding_attention_mask,impression_attention_mask
                                                                                  ,layer_id, self.decoder_F, self.D_F)
                D_loss_l, G_loss_l, pre_image_l, image_l = self.Loss_on_layer_GAN(image_l, finding_input_ids, impression_input_ids,finding_attention_mask, impression_attention_mask, 
                                                                                  layer_id, self.decoder_L, self.D_L)

                # # train with view consistency loss
                # self.G_optimizer.zero_grad()
                # pred = self.embednet(pre_image_f, pre_image_l)
                # id_loss = 1 * self.S_criterion(pred, torch.zeros_like(pred).to(self.device))
                # id_loss.backward(retain_graph='True')
                # self.G_optimizer.step()

                if ((idx + 1) % DISP_FREQ == 0) and idx != 0:
                    # ...log the running loss
                    # self.writer.add_scalar("Train_{}_SSIM".format(layer_id), ssim.ssim(r_image, pre_image).item(),
                    #                        epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('GAN_G_train_Layer_front_{}_loss'.format(layer_id),
                                           G_loss_f.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('GAN_D_train_Layer_front_{}_loss'.format(layer_id),
                                           D_loss_f.item(),
                                           epoch * len(self.train_dataloader) + idx)

                    self.writer.add_scalar('GAN_G_train_Layer_lateral_{}_loss'.format(layer_id),
                                           G_loss_l.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_scalar('GAN_D_train_Layer_lateral_{}_loss'.format(layer_id),
                                           D_loss_l.item(),
                                           epoch * len(self.train_dataloader) + idx)
                    # write to tensorboard
                    self.writer.add_images("GAN_Train_Original_front_{}".format(layer_id),
                                           (image_f + 1) / 2,
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("GAN_Train_Predicted_front_{}".format(layer_id),
                                           (pre_image_f + 1) / 2,
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("GAN_Train_Original_lateral_{}".format(layer_id),
                                           (image_l + 1) / 2,
                                           epoch * len(self.train_dataloader) + idx)
                    self.writer.add_images("GAN_Train_Predicted_lateral_{}".format(layer_id),
                                           (pre_image_l + 1) / 2,
                                           epoch * len(self.train_dataloader) + idx)
            
            
            tb=self.writer
            
            self.G_lr_scheduler.step(epoch)
            self.D_lr_scheduler.step(epoch)
            if (epoch +1) % 1== 0 and epoch != 0:
                torch.save(self.encoder.state_dict(), os.path.join(self.encoder_checkpoint,
                                                                   "Encoder_{}_Layer_{}_checkpoint.pth".format(
                                                                       Encoder, layer_id)))
                torch.save(self.D_F.state_dict(), os.path.join(self.D_checkpoint,
                                                               "D_{}_F_Layer_{}_checkpoint.pth".format(
                                                                   Discriminator, layer_id)))
                torch.save(self.D_L.state_dict(), os.path.join(self.D_checkpoint,
                                                               "D_{}_L_Layer_{}_checkpoint.pth".format(
                                                                   Discriminator, layer_id)))

                torch.save(self.decoder_F.state_dict(), os.path.join(self.decoder_checkpoint,
                                                                     "Decoder_{}_F_Layer_{}_checkpoint.pth".format(
                                                                         Decoder, layer_id)))

                torch.save(self.decoder_L.state_dict(), os.path.join(self.decoder_checkpoint,
                                                                     "Decoder_{}_L_Layer_{}_checkpoint.pth".format(
                                                                         Decoder, layer_id)))

    def train(self):

        # self.load_model()

        for layer_id in range(self.P_ratio + 1):

            self.define_D(layer_id)
            # self.define_dataloader(layer_id)
            self.define_opt(layer_id)

            # Train VCN by layer

            #print(f"Start training on Siamese {layer_id}")

            # self.train_Siamese_layer(layer_id)
            # Train Generator by layer

            if layer_id == 0:
                print("Start training on Decoder {}".format(layer_id))
                self.train_layer(layer_id)

            # Train GAN by layer

            print("Start training GAN {}".format(layer_id))
            self.train_GAN_layer(layer_id)

        def pare_cfg(cfg_json):
            with open(cfg_json) as f:
                cfg = f.read()
                print(cfg)
                print("Config Loaded")
            return json.loads(cfg)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
