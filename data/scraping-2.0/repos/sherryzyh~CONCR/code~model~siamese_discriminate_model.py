import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
from transformers import OpenAIGPTConfig, OpenAIGPTModel, XLNetConfig, XLNetModel, GPT2Model, GPT2Config
from transformers import BartConfig, BartForSequenceClassification, BartModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from .discriminate_model import pretrained_model
from .scorers import CosSimilarity, CausalScorer


class siamese_reasoning_model(nn.Module):
    def __init__(self, hps):
        super(siamese_reasoning_model, self).__init__()
        self.hps = hps
        self.model_name = hps.model_name
        self.model_type = "contrastive"

        if hps.model_name == 'bert':
            self.sentence_encoder = BertModel.from_pretrained(hps.model_dir)
            self.config = BertConfig(hps.model_dir)
        elif hps.model_name == 'roberta':
            self.sentence_encoder = RobertaModel.from_pretrained(hps.model_dir)
            self.config = RobertaConfig(hps.model_dir)
        elif hps.model_name == 'xlnet':
            self.sentence_encoder = XLNetModel.from_pretrained(hps.model_dir, mem_len=1024)
            self.config = XLNetConfig.from_pretrained(hps.model_dir)
        elif hps.model_name == 'bart':
            self.sentence_encoder = BartModel.from_pretrained(hps.model_dir)
            self.config = BartConfig.from_pretrained(hps.model_dir)
        elif hps.model_name == 'gpt2':
            self.sentence_encoder = GPT2Model.from_pretrained(hps.model_dir)
            self.config = GPT2Config.from_pretrained(hps.model_dir)

        # Scorer
        if hps.score == "cossim":
            self.sim = CosSimilarity()
        elif hps.score == "causalscore":
            self.sim = CausalScorer(self.config)

    # compose causal pairs
    def compose_causal_pair(self, premise, hypothesis, ask_for, device):
        """
        Arguments:
            premise     [batch_size, hidden_size]
            hypothesis  [batch_size, hidden_size]
            ask_for     [batch_size]
            device
        """
        batch_size = ask_for.size(0)
        causes = torch.zeros((batch_size, self.config.hidden_size))
        effects = torch.zeros((batch_size, self.config.hidden_size))

        for i in range(batch_size):
            if ask_for[0] == 0:  # 'ask-for cause'
                causes[i] = hypothesis[i]
                effects[i] = premise[i]
            else:  # 'ask-for effect'
                causes[i] = premise[i]
                effects[i] = hypothesis[i]
        return causes.to(device), effects.to(device)

    # forward sentence embedding
    def forward_sent_encoding(self, input_ids, attention_mask, seg_ids=None, length=None):
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)

        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if seg_ids is not None:
            seg_ids = seg_ids.view((-1, seg_ids.size(-1)))
        if length is not None:
            length = length.view(-1)

        if self.hps.model_name in ['bert', 'albert', 'gpt2']:
            sent_embs = self.sentence_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                              token_type_ids=seg_ids)
        else:
            sent_embs = self.sentence_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Pooling
        # by default, use the "cls" embedding as the sentence representation
        if self.hps.model_name in ["bert", "roberta", "albert"]:
            pooler_output = sent_embs.pooler_output
        elif self.hps.model_name in ["xlnet"]:
            pooler_output = sent_embs.last_hidden_state[:, 0, :]
        elif self.hps.model_name == "bart":
            pooler_output = sent_embs.encoder_last_hidden_state[:, 0, :]
        elif self.hps.model_name == "gpt2":
            last_hidden_state = sent_embs.last_hidden_state
            pooler_output = torch.nanmean(last_hidden_state, dim=1)

        # print("pooler_output.size:", pooler_output.size())
        pooler_output = pooler_output.view(
            (batch_size, num_sent, pooler_output.size(-1)))  # [bs, num_sent, hidden_size]
        return pooler_output

    def forward(self, input_ids, attention_mask, ask_for, seg_ids=None, length=None, mode='train'):
        if mode == "train":
            return self.train_forward(input_ids, attention_mask, ask_for, seg_ids, length)
        elif mode == "eval":
            return self.eval_forward(input_ids, attention_mask, ask_for, seg_ids, length)

    def train_forward(self, input_ids, attention_mask, ask_for, seg_ids=None, length=None):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Sentence pooler encoding
        pooler_output = self.forward_sent_encoding(input_ids, attention_mask, seg_ids, length)

        # Cause/Effect representation alignment
        premise, hypothesis = pooler_output[:, 0], pooler_output[:, 1]
        causes, effects = self.compose_causal_pair(premise, hypothesis, ask_for, device)
        scores_matirx = self.sim(causes, effects)
        scores = torch.diagonal(scores_matirx, offset=0).unsqueeze(1)
        return scores

    def eval_forward(self, input_ids, attention_mask, ask_for, seg_ids=None, length=None):
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        device = input_ids.device

        # Sentence pooler encoding
        pooler_output = self.forward_sent_encoding(input_ids, attention_mask, seg_ids, length)  # [bs, num_sent, hidden_size]


        # Separate representation
        premise, hypothesis_0, hypothesis_1 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]

        # Scoring hypothesis 0
        causes_0, effects_0 = self.compose_causal_pair(premise, hypothesis_0, ask_for, device)
        premise_0_score_matrix = self.sim(causes_0, effects_0)
        score_0 = torch.diagonal(premise_0_score_matrix, offset=0).unsqueeze(1)

        # Scoring hypothesis 1
        causes_1, effects_1 = self.compose_causal_pair(premise, hypothesis_1, ask_for, device)
        premise_1_score_matrix = self.sim(causes_1, effects_1)
        score_1 = torch.diagonal(premise_1_score_matrix, offset=0).unsqueeze(1)

        # score = torch.cat([score_0, score_1], dim=1)
        return score_0, score_1