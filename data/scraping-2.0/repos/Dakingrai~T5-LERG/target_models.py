import torch
import torch.nn.functional as F
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
import pdb

def get_sum_multi_head_attentions(multi_head_attentions):
    return sum(torch.sum(x,1) for x in multi_head_attentions)

class GPT:
    def __init__(self, model_dir="models/dailydialog_gpt", device="cuda" if torch.cuda.is_available() else "cpu", evaluate=False):
        self.device = device
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(model_dir)
        self.model = OpenAIGPTLMHeadModel.from_pretrained(model_dir, output_attentions=True).to(device)
        self.SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
        self.bos, self.eos, self.speaker1, self.speaker2, self.padding = \
            self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)
    def forward(self, inputs, label=None, is_x_tokenized=False, is_y_tokenized=False, output_type="prob"):
        x_set = [self.tokenizer.convert_tokens_to_ids(x) for x in inputs] if is_x_tokenized \
            else [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x)) for x in inputs]
        x_set = [[self.speaker1] + x for x in x_set]
        max_l = max(len(x) for x in x_set)
        x_set = [x + [self.padding] * (max_l - len(x)) for x in x_set]
        y = self.tokenizer.convert_tokens_to_ids(label) if is_y_tokenized \
            else self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label))
        y = [self.speaker2] + y
        
        input_ids = torch.tensor([[self.bos] + x+y for x in x_set]).to(self.device)
        token_type_ids = torch.tensor([[self.speaker1] * (len(x)+1) + [self.speaker2] * len(y) for x in x_set]).to(self.device)

        if output_type == "prob":
            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=token_type_ids)
                probs = F.softmax(outputs.logits, dim=-1)
            return probs[:,-len(y):-1,:], y[1:]
        elif output_type == "attn":
            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=token_type_ids)
                probs = F.softmax(outputs.logits, dim=-1)
            attn = get_sum_multi_head_attentions(outputs[-1])
            attn = attn[0][max_l+1:, 1:max_l]
            attn_map = {}
            for xi in range(max_l-1):
                for yi in range(len(y)-1):
                    attn_map[(xi,yi)] = attn[yi,xi]
            attn_set = torch.sum(attn,dim=0)
            return attn_set, attn_map, self.tokenizer.tokenize(inputs[0]), self.tokenizer.tokenize(label)
        elif output_type == "grad":
            count = 0
            for param in self.model.parameters():
                if count > 0:
                    param.requires_grad = False
                else:
                    embeds = param
                count +=1
            input_ids = torch.tensor([[self.bos] + x+y + [self.eos] for x in x_set]).to(self.device)
            token_type_ids = torch.tensor([[self.speaker1] * (len(x)+1) + [self.speaker2] * (len(y)+1) for x in x_set]).to(self.device)
            outputs = self.model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
            losses = F.cross_entropy(outputs.logits[0,max_l+2:-1,:], input_ids[0,max_l+3:], reduction="none")
            scores = []
            for j in range(len(y)-1):
                grads = torch.autograd.grad(losses[j],embeds,retain_graph=True, create_graph=False)[0]
                mod = embeds - grads
                changes = torch.norm(mod, dim=1) - torch.norm(embeds, dim=1)
                scores.append(changes[input_ids[0,2:max_l+1]])
            grad_map = {}
            for xi in range(max_l-1):
                for yi in range(len(y)-1):
                    grad_map[(xi,yi)] = scores[yi][xi]
            grad_set = torch.sum(torch.stack(scores),dim=0)
            return grad_set, grad_map, self.tokenizer.tokenize(inputs[0]), self.tokenizer.tokenize(label)
        

class T5:
    def __init__(self, model_dir="models/checkpoint-16000", device="cuda" if torch.cuda.is_available() else "cpu", evaluate=False):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
        self.model.eval()

    def forward(self, inputs, label=None, is_x_tokenized=False, is_y_tokenized=False, output_type="prob"):
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in inputs]) if is_x_tokenized \
            else  torch.tensor([self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x)) for x in inputs])
        attention_mask =  torch.tensor([[1] * len(x) for x in input_ids])
        label_id = self.tokenizer.convert_tokens_to_ids(label) if is_y_tokenized \
            else self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label))
        label_id = torch.tensor([label_id for each in range(len(inputs))])

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device), 
                attention_mask=attention_mask.to(self.device), 
                labels=label_id.to(self.device))
            
        probs = F.softmax(outputs.logits, dim=-1)[:]
        # output_text = self.tokenizer.convert_ids_to_tokens(torch.argmax(probs, dim=-1)[0])
        return probs, label_id[0]