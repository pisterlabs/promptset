# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from util import mylog
from modules.until_module import PreTrainedModel,  CrossEn, AllGather, Loss_recoder
from modules.module_audio import AudioCLIP
from modules.module_clip import MultiLingualCLIP, convert_weights




coffient = 100.0
logger = mylog().getlog()#logging.getLogger(__name__)
allgather = AllGather.apply

def show_log(task_config, info):
    
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

class MCLIP4VLAPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self,*inputs, **kwargs):
        super(MCLIP4VLAPreTrainedModel, self).__init__()
        
        self.multilingualclip = None
        self.audio = None
     

    @classmethod
    def from_pretrained(cls, state_dict=None, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        pretrained_clip_name = "RN50x16"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        
        clip_state_dict = MultiLingualCLIP.get_config(pretrained_clip_name=pretrained_clip_name)

        
        ###############################################
        pre_trained_model = ''
        if state_dict is None: 
            state_dict = {}
            
            if clip_state_dict is not None:
                pre_trained_model += 'clip/'+ pretrained_clip_name   
        else:
            pre_trained_model = 'initial_model'
          
        model = cls(clip_state_dict,  *inputs, **kwargs)

        assert model.multilingualclip is not None
        model = cls.init_preweight(model, state_dict, task_config=task_config, pre_trained_model=pre_trained_model)
        

        ## ####################################
        # freeze layers
        ## ####################################
        
        assert task_config.freeze_layer_num <= 13 and task_config.freeze_layer_num >= -1
        if task_config.freeze_layer_num == 13:
            show_log(task_config, "Freeze all clip params. ")
        elif task_config.freeze_layer_num == -1:
            show_log(task_config, "Training all clip params. ")
           

        if hasattr(model, "clip") and task_config.freeze_layer_num > -1:
            for name, param in model.clip.named_parameters():  
                if task_config.freeze_layer_num == 13:
                    param.requires_grad= False
                    continue
                
                if task_config.freeze is not None and name.find(task_config.freeze)==0:
                    param.requires_grad= False
                    show_log(task_config, "Freeze Parameter clip.{} ".format(name))
                    continue    # need to train
                # top layers always need to train
                if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                        or name.find("visual.attnpool") == 0 or name.find("visual.proj") == 0:
                    show_log(task_config, "Training Parameter clip.{} ".format(name))
                    continue    # need to train
                elif name.find("visual.transformer.resblocks.") == 0 or  name.find("visual.layer") == 0 or name.find("transformer.resblocks") == 0:
                    if name.find("visual.layer") == 0:
                        layer_num = int(name.split("visual.layer")[1].split(".")[0])
                    else:
                        layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                    if layer_num >= task_config.freeze_layer_num:
                        show_log(task_config, "Training Parameter clip.{}  ".format(name))
                        continue    # need to train

                
                    # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False
                show_log(task_config, "Freezed Parameter clip.{} ".format(name))

        if task_config.freeze is not None:
             for name, param in model.named_parameters():
                 if name.find(task_config.freeze)==0:
                      param.requires_grad= False

        num_params_total = sum(p.numel() for p in model.parameters())
        num_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
        if num_params_total > 1e6:
            num_params_total /= 1e6
            num_params_train /= 1e6
            params_total_label = 'M'
        elif num_params_total > 1e3:
            num_params_total /= 1e3
            num_params_train /= 1e3
            params_total_label = 'k'
        show_log(task_config,"Total Parameters:{:.2f}{}".format(num_params_total,params_total_label))
        show_log(task_config,"Total Training Parameters:{:.2f}{}".format(num_params_train,params_total_label))     

        return model



def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class MCLIP4VLA(MCLIP4VLAPreTrainedModel):
    
    def __init__(self,  clip_state_dict, task_config):
        super(MCLIP4VLA, self).__init__( clip_state_dict)
        self.task_config = task_config
      
         # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = counts #len([k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".ln_2.weight")])
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        self.vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        
        transformer_heads = transformer_width // 64
        # transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(self.vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        # show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))
        show_log(self.task_config,"\t loss_type:{}".format(self.task_config.loss_func))

      
        self.gradient_checkpoint = False
        if check_attr("gradient_checkpoint", self.task_config):
            self.gradient_checkpoint = True
            show_log(task_config, "\t gradient_checkpoint: {}".format(self.gradient_checkpoint))
        

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.multilingualclip = MultiLingualCLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, self.vocab_size, transformer_width,
            
        ).float()
        
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        if self.task_config.fp16 == False:
            convert_weights(self.multilingualclip.visual)
        # <=== End of CLIP Encoders


        # Audio Encoder ===>
        self.audio = AudioCLIP(embed_dim=embed_dim,
        image_resolution = image_resolution, vision_layers = vision_layers, \
            vision_width = vision_width, vision_patch_size = vision_patch_size, \
                with_control_token=self.task_config.with_control_token).float()
        if self.task_config.fp16 == False:
            convert_weights(self.audio)
        # <=== End of Audio Encoder

        '''
        mil-NCE loss for joint matching(not cross matching)
        '''
        self.nceloss = CrossEn() 
        self.apply(self.init_weights)

    def forward(self, input_txt=None, attention_mask=None, video=None, video_mask=None,
                audio=None, audio_mask=None, input_ids=None, bg_token_gt=None):
        
        if input_txt is not None:
            
            sequence_output = self.get_sequence_output(input_txt,attention_mask)
        else:
            sequence_output = None
            attention_mask = None
    
                                                                                                      
        # attention_mask: text token mask
        if video is not None and video_mask is not None:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            
            visual_output = self.get_visual_output(video, video_mask,  shaped=True)
            
        else:
            visual_output = None
            video_mask = None
      
        if audio is not None and audio_mask is not None:
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            audio = torch.as_tensor(audio).float()
            b, pair, bs  = audio.shape[:3]
            audio = audio.view(b * pair * bs , *audio.shape[3:]) 
            audio_output, _ = self.get_audio_output(audio, audio_mask, token_type=bg_token_gt, shaped=True)
        else:
            audio_output = None
            audio_mask = None
        '''
        sequence_output:[batchsize, max_word_len, 512]
        visual_output:[batchsize, max_frame_len, 512]
        audio_output:[batchsize, max_audio_len, 512]
        '''
        if self.training:
            loss_recoder = Loss_recoder()
            loss = 0.
             
            if visual_output is None and video_mask is None:  
                sim_matrix_t_a = self.get_similarity_logits(sequence_output, audio_output, attention_mask, audio_mask,'text','audio',input_ids=input_ids, shaped=True)     

                sim_loss_t_a = self.nceloss(sim_matrix_t_a)
                
                loss_recoder.update('ta_nce', sim_loss_t_a)
                loss+=sim_loss_t_a

            elif audio_output is None and audio_mask is None:
                sim_matrix_t_v = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,'text','video', input_ids=input_ids,shaped=True)
                        
                sim_loss_t_v = self.nceloss(sim_matrix_t_v) 
                loss_recoder.update('tv_nce',sim_loss_t_v)
                loss+=sim_loss_t_v

            else:
                sim_dict= self.get_similarity_logits_with_three_modalities(sequence_output, visual_output, audio_output, \
                    attention_mask, video_mask, audio_mask,input_ids=input_ids, shaped=True)
                        
                      
                if  'nce' in self.task_config.loss_func:
                    sim_loss_t_v_nce = self.nceloss(sim_dict['t_v']) 
                    sim_loss_t_a_nce = self.nceloss(sim_dict['t_a']) 
                    sim_loss_v_a_nce = (self.nceloss(sim_dict['a_v']) + self.nceloss(sim_dict['a_v'].T)) / 2 
                    sim_loss_t_av_nce = self.nceloss(sim_dict['t_va'])
                            
                    loss_recoder.update('va_nce', sim_loss_v_a_nce)
                    loss_recoder.update('ta_nce', sim_loss_t_a_nce)
                    loss_recoder.update('tv_nce', sim_loss_t_v_nce)
                    loss_recoder.update('tav_nce', sim_loss_t_av_nce)
                    if self.task_config.loss_func == 'ta_nce':
                        loss += sim_loss_t_a_nce 
                    elif self.task_config.loss_func == 'tv_nce':
                        loss += sim_loss_t_v_nce 
                    elif self.task_config.loss_func == 'av_nce':
                        loss += sim_loss_v_a_nce
                    elif self.task_config.loss_func == 'tav_nce':
                        loss += sim_loss_t_av_nce
                    else:
                        raise NotImplementedError

            return loss, loss_recoder
        else:
            return None

    def get_visual_output(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair = video_mask.size(0)
        
        if self.task_config.pretrained_clip_name.startswith('RN'):
            visual_hidden = self.multilingualclip.encode_image_resnet(video).float()
        else:
            visual_hidden = self.multilingualclip.encode_image_transformer(video).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        
        return visual_hidden

    def get_sequence_output(self, input_txt, attention_mask):
        sequence_hidden= self.multilingualclip.encode_text(input_txt, attention_mask)
        return sequence_hidden
    
    def get_audio_output(self, audio, audio_mask, token_type=None,shaped=False):
        if shaped is False:
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            
            b, pair, ts = audio.shape[:3]
            audio = audio.view(b * pair*ts , *audio.shape[3:])

        if token_type is not None:
            token_type = token_type.expand(audio_mask.shape).reshape(-1).to(torch.int64)    

        bs_pair = audio_mask.size(0)
        audio_hidden, bg_hidden  = self.audio(audio, token_type = token_type)
        audio_hidden = audio_hidden.view(bs_pair, -1, audio_hidden.size(-1))
        if bg_hidden is not None:
            bg_hidden = bg_hidden.view(bs_pair, -1, audio_hidden.size(-1))
        
        return audio_hidden, bg_hidden
               
    def _mean_pooling_for_single_modal(self, modal_output, modal_mask, modal_type):
        assert modal_type in ['video', 'text', 'audio']
        if modal_type == 'text' :# add an [cls] token
            modal_mask_un = modal_mask.to(dtype=torch.float).unsqueeze(-1)#[batchsize, max_text_len, 1]
            modal_mask_un[:, 0, :] = 0. #[cls] token
            modal_output = modal_output * modal_mask_un
            modal_mask_un_sum = torch.sum(modal_mask_un, dim=1, dtype=torch.float)
            modal_mask_un_sum = modal_mask_un_sum + torch.ones_like(modal_mask_un_sum, dtype=torch.float)*1e-10
            modal_out = torch.sum(modal_output, dim=1) / modal_mask_un_sum 
        #[batchsize, text_dim]
        elif modal_type == 'video' or modal_type == 'audio':
            if modal_output.shape[1] == modal_mask.shape[1]:
                modal_mask_un = modal_mask.to(dtype=torch.float).unsqueeze(-1)#[batchsize, max_frame_len, 1]
                modal_output = modal_output * modal_mask_un
                modal_mask_un_sum = torch.sum(modal_mask_un, dim=1, dtype=torch.float)
                modal_mask_un_sum = modal_mask_un_sum + torch.ones_like(modal_mask_un_sum, dtype=torch.float)*1e-10
                modal_out = torch.sum(modal_output, dim=1) / modal_mask_un_sum 
            else:
                modal_out = modal_output.mean(dim=1)
        #[batchsize, frame_dim]
        
        return modal_out

    def get_similarity_logits_with_three_modalities(self, sequence_output, visual_output, audio_output, attention_mask, video_mask, audio_mask=None, shaped = False, _pretrain_joint=False, concat_mask=None, input_ids=None, hard_negative=False): 
        
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            audio_mask = audio_mask.view(-1, audio_mask.shape[-1])
            if input_ids is not None:
                input_ids = input_ids.view(-1, input_ids.shape[-1])
    
        # if sequence_output.dim() == 3 and  sequence_output.shape[1]>1:
        #     sequence_output = sequence_output[torch.arange(sequence_output.shape[0]), (input_ids==END_TOKEN).nonzero(as_tuple=True)[1]]
        #     sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        # else:
        #     if sequence_output.dim() == 3 and  sequence_output.shape[1]==1:
        #         sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        
        visual_output = visual_output / (visual_output.norm(dim=-1, keepdim=True)+1e-10)
        visual_output = self._mean_pooling_for_single_modal(visual_output, video_mask, 'video')
        visual_output = visual_output / (visual_output.norm(dim=-1, keepdim=True) + 1e-10)

        audio_output = audio_output / (audio_output.norm(dim=-1, keepdim=True)+1e-10)   
        audio_output = self._mean_pooling_for_single_modal(audio_output, audio_mask, 'audio')
        audio_output = audio_output / (audio_output.norm(dim=-1, keepdim=True) + 1e-10)
            
        sim_matrix_t_v = torch.matmul(sequence_output, visual_output.t()) * coffient
        sim_matrix_t_a = torch.matmul(sequence_output, audio_output.t()) * coffient
        sim_matrix_v_a = torch.matmul(visual_output, audio_output.t()) * coffient

        a_v_output = (visual_output + audio_output)/2
        query_weights = torch.ones((sequence_output.shape[0], visual_output.shape[0], 2)) * 0.5
        sim_matrix_t_av = torch.matmul(sequence_output, a_v_output.t()) * coffient
                        
        retrieval_logits={
            't_v':sim_matrix_t_v,
            't_a':sim_matrix_t_a,
            'a_v':sim_matrix_v_a,
            't_va':sim_matrix_t_av,
            'query_weights':query_weights
        }
        
        return retrieval_logits

    def get_similarity_logits(self, modal1_output, modal2_output, modal1_mask, modal2_mask, modal1='text', modal2='video', shaped=False, _input_ids=None):
        '''
        MIL-NCE loss of text sequence and video sequence.
        sequence_output:[batchsize, max_text_len, text_dim=768]
        visual_output:[batchsize, max_frame_len, visual_dim=768]
        attention_mask:[batchsize, max_text_len]
        video_mask:[batchsize, max_frame_len]
        '''

        if shaped is False:
            modal1_mask = modal1_mask.view(-1, modal1_mask.shape[-1])
            modal2_mask = modal2_mask.view(-1, modal2_mask.shape[-1])
            if input_ids is not None:
                input_ids = input_ids.view(-1, input_ids.shape[-1])

        # if modal1_output.dim() == 3 and  modal1_output.shape[1]>1:
        #     modal1_output = modal1_output[torch.arange(modal1_output.shape[0]), (input_ids==END_TOKEN).nonzero(as_tuple=True)[1]] 
        #     modal1_output = modal1_output / (modal1_output.norm(dim=-1, keepdim=True) + 1e-10)
        # else:
        #     if modal1_output.dim() == 3 and  modal1_output.shape[1]==1:
        #         modal1_output = modal1_output.squeeze(1)
        modal1_output = modal1_output / (modal1_output.norm(dim=-1, keepdim=True) + 1e-10)

        modal2_output = modal2_output / (modal2_output.norm(dim=-1, keepdim=True) + 1e-10)
        modal2_output = self._mean_pooling_for_single_modal(modal2_output, modal2_mask, modal2)
        modal2_output = modal2_output / (modal2_output.norm(dim=-1, keepdim=True) + 1e-10)
        sim_matrix=torch.matmul(modal1_output, modal2_output.t())*coffient
            
        return sim_matrix

