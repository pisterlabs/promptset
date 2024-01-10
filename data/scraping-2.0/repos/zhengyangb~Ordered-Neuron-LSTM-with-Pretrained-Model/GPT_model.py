import torch
import torch.nn as nn
import pdb

from embed_regularize import embedded_dropout_gpt
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from ON_LSTM import ONLSTMStack
from GPT.pytorch_pretrained_bert import OpenAIGPTModel, OpenAIGPTConfig, OpenAIGPTLMHead


class GPTRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, chunk_size, nlayers,  dropout=0.5, dropouth=0.5, dropouti=0.5,
                 dropoute=0.1, wdrop=0, tie_weights=False, args=None):
        super(GPTRNNModel, self).__init__()
        self.transformer = OpenAIGPTModel.from_pretrained('openai-gpt')
        config = OpenAIGPTConfig()
        self.lm_head = OpenAIGPTLMHead(self.transformer.tokens_embed.weight, config)
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(768, ninp)
        self.args = args

        assert rnn_type in ['LSTM'], 'RNN type is not supported'
        self.rnn = ONLSTMStack(
            [ninp] + [nhid] * (nlayers - 1) + [ninp],
            chunk_size=chunk_size,
            dropconnect=wdrop,
            dropout=dropouth
        )
        self.decoder = nn.Linear(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     #if nhid != ninp:
        #     #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.distance = None
        self.tie_weights = tie_weights



    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self, pre_emb):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # if pre_emb is not None:
        #     self.encoder.weight.data[:pre_emb.size(0), :pre_emb.size(1)] = torch.FloatTensor(pre_emb)

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, gpt_ids, fl_ids, return_h=False):
        if self.args.feature is not None and 'fixGPT' in self.args.feature.split('_'):
            with torch.no_grad():
                emb = self.transformer(gpt_ids)
        else:
            emb = self.transformer(gpt_ids)  # BS * GPT_SL * GPT_EMS

        lm_logits = self.lm_head(emb)
        # shift_logits = lm_logits[..., :-1, :].contiguous()
        emb = torch.cat([emb[r:r+1, fl_ids[r], :] for r in range(len(fl_ids))], dim=0)  # BS * (2*SL) * GPT_ES
        emb = torch.nn.functional.avg_pool1d(emb.permute(0, 2, 1), 2) * 2  # BS * GPT_EMS * SL
        emb = emb.permute(2, 0, 1)  # BS * SL * GPT_EMS -> SL * BS * ES
        self.encoder = embedded_dropout_gpt(self.encoder,
            dropout=self.dropoute if self.training else 0
        )
        emb = nn.functional.relu(self.encoder(emb))
        emb = self.lockdrop(emb, self.dropouti)

        raw_output, hidden, raw_outputs, outputs, self.distance = self.rnn(emb, hidden)

        output = self.lockdrop(raw_output, self.dropout)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs, lm_logits.view(-1, lm_logits.size(-1))
        else:
            return result, hidden

    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)
