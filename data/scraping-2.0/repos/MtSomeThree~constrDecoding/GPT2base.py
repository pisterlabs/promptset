import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel, OpenAIGPTPreTrainedModel, OpenAIGPTConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutput

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn.functional import softmax
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, Sigmoid, LogSigmoid

import numpy as np

class ConstrainedLM(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_rc = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.model_rc_transformer = GPT2Model.from_pretrained("gpt2")
        self.constraint_factor = 1.0
        self.temperature = 1.0
        self.use_rc_transformer = False
        self.log_sigmoid_fct = LogSigmoid()

    def set_model_rc_transformer(self, model_rc_transformer):
        self.model_rc_transformer = model_rc_transformer

    def set_constraint_factor(self, factor):
        self.constraint_factor = factor

    def set_use_rc_transformer(self, use_rc_transformer):
        self.use_rc_transformer = use_rc_transformer

    def set_temperature(self, temperature):
        self.temperature = temperature

    def get_model_rc(self):
        return self.model_rc

    def set_model_rc(self, new_model_rc):
        self.model_rc = new_model_rc

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        rc_labels=None,
        rc_weights=None,
        use_cache=None,
        use_temperature=False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        #return_dict is always None
        #copy from huggingface docs

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs.last_hidden_state

        # if past_key_values is not None and head_mask is not None:
        #     print ("Shape Check:")
        #     print (input_ids.shape)
        #     print (attention_mask.shape)
        #     print (len(past_key_values))
        #     print (len(past_key_values[0]))
        #     print (past_key_values[0][0].shape)
        #     print (head_mask.shape)
        #     print (head_mask.sum())


        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        if use_temperature:
            lm_logits = lm_logits * self.temperature

        if self.constraint_factor == 0.0:
            pred_logits = lm_logits
        else:
            if self.use_rc_transformer:
                rc_hidden_states = self.model_rc_transformer(
                    input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                ).last_hidden_state
            else:
                rc_hidden_states = hidden_states
            constr_logits = self.log_sigmoid_fct(self.model_rc(rc_hidden_states)) * self.constraint_factor
            pred_logits = lm_logits + constr_logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            # Flatten the tokens
            if rc_labels is not None:
                rc_labels = rc_labels.long()
                shift_logits = shift_logits.index_select(0, torch.where(rc_labels == 1)[0])
                shift_labels = shift_labels.index_select(0, torch.where(rc_labels == 1)[0])

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        elif rc_labels is not None:
            length = input_ids.shape[1]
            if len(rc_labels.shape) == 1:
                rc_labels = rc_labels.unsqueeze(1)
            for i in range(length - 1):
                pred_logits = torch.gather(constr_logits[:, i, :], 1, input_ids[:, i + 1].unsqueeze(1))

                if rc_weights is not None:
                    weights = softmax(rc_weights * (1.0 - self.temperature), dim=0)
                    loss_fct = BCEWithLogitsLoss(weight=
                        (attention_mask[:, i + 1] * weights).unsqueeze(1), reduction='sum')
                else:
                    loss_fct = BCEWithLogitsLoss(weight=attention_mask[:, i + 1].unsqueeze(1))

                cur_loss = loss_fct(pred_logits, rc_labels)
                if loss is not None:
                    loss += cur_loss
                else:
                    loss = cur_loss

        return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=pred_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = ConstrainedLM.from_pretrained("gpt2")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))

    sentence_prefix = "I play piano every"

    input_ids = tokenizer.encode(
        sentence_prefix,
        add_special_tokens=False,
        return_tensors="pt",
    )

    #model(input_ids=input_ids)

    for i in range(5):
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_length=30,  # desired output sentence length
            pad_token_id=model.config.eos_token_id,
        )[0].tolist()
         
        generated_text = tokenizer.decode(
            output_ids,
            clean_up_tokenization_spaces=True)
         
        print(generated_text)

