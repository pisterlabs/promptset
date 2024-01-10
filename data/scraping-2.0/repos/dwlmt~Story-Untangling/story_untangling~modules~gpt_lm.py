from typing import Dict, Any, List

import torch
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.openai_transformer import OpenaiTransformer
from torch import nn
from torch.nn import NLLLoss


class BaseLMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, transformer: OpenaiTransformer, metrics: Dict[str, Any] = None,
                 accuracy_top_k: List = None):
        super(BaseLMHead, self).__init__()
        self.transformer = transformer
        self._metrics = metrics
        self._accuracy_top_k = accuracy_top_k

        self._decoder = self.transformer.decoder

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = NLLLoss(ignore_index=0)

    def calc_loss(self, lm_labels, lm_logits):
        # Shift so that tokens < n predict n
        lm_logits = lm_logits[:, :-1, :]
        lm_labels = lm_labels[:, 1:]
        # Flatten the tokens and classes
        lm_logits = lm_logits.contiguous().view(-1, lm_logits.size(-1))
        lm_labels = lm_labels.contiguous().view(-1)
        scores_softmax = self.log_softmax(lm_logits)
        loss = self.loss(scores_softmax, lm_labels)
        if self._metrics and not self.training:
            with torch.no_grad():
                for top_k in self._accuracy_top_k:
                    self._metrics[f"gen_accuracy_{top_k}"](scores_softmax, lm_labels)
                    self._metrics["accuracy_combined"](self._metrics[f"gen_accuracy_{top_k}"].get_metric())
        return loss


class MixtureLM(BaseLMHead):
    """ Language Model mixture for the Transformer. """

    def __init__(self, transformer: OpenaiTransformer, feature_dim: int, metrics: Dict[str, Any] = None,
                 accuracy_top_k: List = None):
        super(MixtureLM, self).__init__(transformer, metrics, accuracy_top_k)

        # Trainable weight for combining the language model.
        self._lm_weighting = torch.tensor([0.5], requires_grad=True)

        self._feature_decoder = torch.nn.Linear(in_features=feature_dim,
                                                out_features=self.transformer.decoder.out_features, bias=False)

    def forward(self, lm_hidden_states, feature_hidden_states, lm_labels=None):

        lm_weighting = self._lm_weighting.clamp(min=0.001, max=0.999).to(feature_hidden_states.device)

        feature_logits = self._feature_decoder(feature_hidden_states)

        if len(lm_hidden_states.shape) == 3:
            feature_logits = feature_logits.unsqueeze(dim=1)

        lm_logits = self._decoder(lm_hidden_states.to(self._decoder.weight.device)).to(feature_logits.device)
        lm_logits = (lm_logits * lm_weighting) + (feature_logits * (1.0 - lm_weighting))

        if lm_labels is not None:
            return self.calc_loss(lm_labels.to(lm_logits.device), lm_logits)

        return lm_logits


class FusionLM(BaseLMHead):
    """ Fuses Language Model output using a Seq2SeqEncoder"""

    def __init__(self, transformer: OpenaiTransformer, encoder: Seq2SeqEncoder, metrics: Dict[str, Any] = None,
                 accuracy_top_k: List = None):
        super(FusionLM, self).__init__(transformer, metrics, accuracy_top_k)

        self._decoder.requires_grad = True

        self._encoder = encoder

    def forward(self, lm_hidden_states, feature_hidden_states, lm_labels=None):

        if len(feature_hidden_states.shape) < len(lm_hidden_states.shape):
            feature_hidden_states = feature_hidden_states.unsqueeze(dim=1)

            feature_hidden_states = feature_hidden_states.expand(feature_hidden_states.shape[0],
                                                                 lm_hidden_states.shape[1],
                                                                 feature_hidden_states.shape[2])

        fused_states = torch.cat((lm_hidden_states, feature_hidden_states), dim=-1)

        fused_states = self._encoder(fused_states)

        lm_logits = self._decoder(fused_states.to(self._decoder.weight.device))

        if lm_labels is not None:
            return self.calc_loss(lm_labels.to(feature_hidden_states.device), lm_logits.to(feature_hidden_states)).to(
                lm_hidden_states.device)

        return lm_logits


class BilinearLM(BaseLMHead):
    """ Fuses Language Model output using a Bilinear layer"""

    def __init__(self, transformer: OpenaiTransformer, feature_dim: int, metrics: Dict[str, Any] = None,
                 accuracy_top_k: List = None):
        super(BilinearLM, self).__init__(transformer, metrics, accuracy_top_k)

        self._bilinear = nn.Bilinear(feature_dim, self._decoder.in_features, self._decoder.in_features)

        self._decoder.requires_grad = True

    def forward(self, lm_hidden_states, feature_hidden_states, lm_labels=None):
        feature_hidden_states = feature_hidden_states.unsqueeze(dim=1)

        feature_hidden_states = feature_hidden_states.expand(feature_hidden_states.shape[0], lm_hidden_states.shape[1],
                                                             feature_hidden_states.shape[2])

        fused_states = self._bilinear(feature_hidden_states.contiguous(), lm_hidden_states)

        lm_logits = self._decoder(fused_states)

        if lm_labels is not None:
            return self.calc_loss(lm_labels, lm_logits).to(lm_hidden_states.device)

        return lm_logits
