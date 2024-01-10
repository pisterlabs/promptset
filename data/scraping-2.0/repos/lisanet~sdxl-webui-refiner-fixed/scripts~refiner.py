import torch
from modules import scripts, script_callbacks, devices, sd_models, sd_models_config, shared
import gradio as gr
import sgm.modules.diffusionmodules.discretizer
from sgm.modules.encoders.modules import ConcatTimestepEmbedderND
from safetensors.torch import load_file, load
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from omegaconf import OmegaConf
from sgm.util import (
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
)

class Refiner(scripts.Script):
    def __init__(self):
        super().__init__()
        self.model = None
        self.base = None
        self.model_name = ''
        self.embedder = ConcatTimestepEmbedderND(256)
        self.c_ae = None
        self.uc_ae = None
        
    def title(self):
        return "Refiner"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def build_model(self):
        refiner_config = OmegaConf.load(sd_models_config.config_sdxl_refiner).model.params.network_config
        self.model = instantiate_from_config(refiner_config)
        self.model = get_obj_from_str(OPENAIUNETWRAPPER)(
            self.model, compile_model=False
        ).eval()
        dev = 'mps' if devices.device.type == 'mps' else 'cpu'   # on Apple Silicon mps is more efficient than switch back and forth
        self.model = self.model.to(dev, devices.dtype_unet)
        self.model.train = disabled_train
        self.model.diffusion_model.dtype = devices.dtype_unet
        self.model.conditioning_key = 'crossattn'
        self.model.cond_stage_key = 'txt'
        self.model.parameterization = 'v'
        discretization = sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization()
        self.model.alphas_cumprod = torch.asarray(discretization.alphas_cumprod, device=devices.device, dtype=devices.dtype_unet)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def load_model(self, model_name):
        if not shared.opts.disable_mmap_load_safetensors:
            ckpt = load_file(sd_models.checkpoints_list[model_name].filename)
        else:
            ckpt = load(open(sd_models.checkpoints_list[model_name].filename, 'rb').read())
        model_type = ''
        for key in ckpt.keys():
            if 'conditioner' in key: 
                model_type = 'Refiner'
            if 'input_blocks.7.1.transformer_blocks.4.attn1.to_k.weight' in key:
                model_type = 'Base'
                break
        if model_type != 'Refiner': 
            self.enable = False
            script_callbacks.remove_current_script_callbacks()
            print('\nNot refiner, extension disabled!\n')
            return False
        
        print('\nLoading refiner...\n')
        self.build_model()
            
        state_dict = dict()
        for key in ckpt.keys():
            if 'model.diffusion_model' in key:
                state_dict[key.replace('model.d', 'd')] = ckpt[key].half()
        self.model.load_state_dict(state_dict)
        self.model_name = model_name
        return True
        
    def ui(self, is_img2img):
        with gr.Accordion(label='Refiner', open=False):
            enable = gr.Checkbox(label='Enable Refiner', value=False)
            with gr.Row():
                checkpoint = gr.Dropdown(choices=['None', *sd_models.checkpoints_list.keys()], label='Model')
            
        ui = [enable, checkpoint]
        return ui
    
    def process(self, p, enable, checkpoint):
        if not enable or checkpoint == 'None':
            script_callbacks.remove_current_script_callbacks()
            self.base = None
            self.model_name = ''
            self.model = None
            devices.torch_gc()
            return
        if self.model_name != checkpoint:
            if not self.load_model(checkpoint): return
        self.c_ae = self.embedder(torch.tensor(shared.opts.sdxl_refiner_high_aesthetic_score).unsqueeze(0).to(devices.device).repeat(p.batch_size, 1))
        self.uc_ae = self.embedder(torch.tensor(shared.opts.sdxl_refiner_low_aesthetic_score).unsqueeze(0).to(devices.device).repeat(p.batch_size, 1))
        p.extra_generation_params['Refiner model'] = checkpoint.rsplit('.', 1)[0]
        p.extra_generation_params['Refiner steps'] = '20 %'
 
        
        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            if params.sampling_step > params.total_sampling_steps - int(params.total_sampling_steps * 0.2) - 2:
                params.text_cond['vector'] = torch.cat((params.text_cond['vector'][:, :2304], self.c_ae), 1)
                params.text_uncond['vector'] = torch.cat((params.text_uncond['vector'][:, :2304], self.uc_ae), 1)
                params.text_cond['crossattn'] = params.text_cond['crossattn'][:, :, -1280:]
                params.text_uncond['crossattn'] = params.text_uncond['crossattn'][:, :, -1280:]
                if self.base is None:
                    self.base, self.model = self.switch_model(p, self.base, self.model)
       
        def denoised_callback(params: script_callbacks.CFGDenoiserParams):
            if params.sampling_step == params.total_sampling_steps - 2:
                self.model, self.base = self.switch_model(p, self.model, self.base)
        
        script_callbacks.on_cfg_denoiser(denoiser_callback)
        script_callbacks.on_cfg_denoised(denoised_callback)
    
    def switch_model(self, p, oldmodel, newmodel):
        if devices.device.type == 'mps':     # on Apple Silicon mps is more efficient than switch back and forth
            oldmodel = p.sd_model.model
        else:
            oldmodel = p.sd_model.model.to('cpu', devices.dtype_unet)
            devices.torch_gc()
        p.sd_model.model = newmodel.to(devices.device, devices.dtype_unet)
        newmodel = None
        return oldmodel, newmodel

    def postprocess(self, p, processed, *args):
        if self.base is not None:
            self.model, self.base = self.switch_model(p, self.model, self.base)
        script_callbacks.remove_current_script_callbacks()
        
