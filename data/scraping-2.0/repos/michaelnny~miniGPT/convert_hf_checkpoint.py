"""Loads the GPT-2 model weights from hugging face transformers and convert it state_dict to support our model"""
import os
import torch
from transformers import GPT2LMHeadModel as HF_GPT2LMHeadModel

from models import GPT2LMHeadModel


def convert_weights(model_type: str, save_path: str) -> None:
    """Load weights from openAI pretrained GPT-2 model using transformers,
    and save it as state_dict so it's compatible with our model."""

    assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')

    if os.path.exists(save_path):
        print(f'The checkpoint file {save_path} already exists, aborting...')
        return

    print(f'Loading weights for {model_type} model from transformers...')
    # create a hugging face transformers model
    hf_model = HF_GPT2LMHeadModel.from_pretrained(model_type)
    hf_state_dict = hf_model.state_dict()

    # copy while ensuring all of the parameters are aligned and match in names and shapes
    hf_state_dict_keys = hf_state_dict.keys()
    hf_state_dict_keys = [k for k in hf_state_dict_keys if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
    hf_state_dict_keys = [k for k in hf_state_dict_keys if not k.endswith('.attn.bias')]  # same, just the mask (buffer)

    # the openAI checkpoints use a "Conv1D" module, but we want to use a Linear module
    # this means that we have to transpose these weights when we import them
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    state_dict = {}

    for hf_key in hf_state_dict_keys:
        if any(hf_key.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            params = hf_state_dict[hf_key].t()
        else:
            params = hf_state_dict[hf_key]

        # hugging face model has different module names
        k = (
            hf_key.replace('transformer.wte.', 'transformer.token_embed.')
            .replace('transformer.wpe.', 'transformer.position_embed.')
            .replace('transformer.h.', 'transformer.layers.')
            .replace('.attn.', '.mh_attn.')
            .replace('transformer.ln_f.', 'transformer.post_ln.')
        )

        state_dict[k] = params.clone()

    # verify that we can load the state_dict into our model
    model = GPT2LMHeadModel(model_type)
    model.load_state_dict(state_dict)

    torch.save(state_dict, save_path)

    del model
    del state_dict

    print(f'Checkpoint saved at {save_path}')


if __name__ == '__main__':
    # convert_weights(model_type='gpt2', save_path='./checkpoints/gpt2-openai-pretrained.pt')
    # convert_weights(model_type='gpt2-medium', save_path='./checkpoints/gpt2-medium-openai-pretrained.pt')
    # convert_weights(model_type='gpt2-large', save_path='./checkpoints/gpt2-large-openai-pretrained.pt')
    convert_weights(model_type='gpt2-xl', save_path='./checkpoints/gpt2-xl-openai-pretrained.pt')
