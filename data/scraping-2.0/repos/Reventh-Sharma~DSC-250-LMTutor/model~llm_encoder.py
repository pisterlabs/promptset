from typing import Any, Dict, List, Optional

import torch
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, Field
from loguru import logger
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class LLMBasedEmbeddings(Embeddings):
    def __init__(
        self,
        model,
        tokenizer,
        device,
        aggregation="mean",
        hidden_state_id=-1,
    ):
        """Initialize the encoding model."""
        self.model = model
        self.tokenizer = tokenizer
        self.aggregation = aggregation
        self.device = device
        self.hidden_state_id = hidden_state_id

        logger.info(
            f"Initialized {self.__class__.__name__}. with hidden_state_id: {self.hidden_state_id} and aggregation: {aggregation}"
        )

    @torch.no_grad()
    def embed_documents(
        self, texts: List[str], max_tokens: int = None
    ) -> List[List[float]]:
        """Compute doc embeddings using a LLM model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        # Example Architecture:
        # LlamaModel(
        #   (embed_tokens): Embedding(32000, 4096, padding_idx=0)
        #   (layers): ModuleList(
        #     (0-31): 32 x LlamaDecoderLayer(
        #       (self_attn): LlamaAttention(
        #         (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #         (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #         (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #         (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        #         (rotary_emb): LlamaRotaryEmbedding()
        #       )
        #       (mlp): LlamaMLP(
        #         (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
        #         (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
        #         (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
        #         (act_fn): SiLUActivation()
        #       )
        #       (input_layernorm): LlamaRMSNorm()
        #       (post_attention_layernorm): LlamaRMSNorm()
        #     )
        #   )
        #   (norm): LlamaRMSNorm()
        # )
        # Output shape in each layer: (batch_size, sequence_length, 4096)
        # There are 33 layers in total, including the embedding layer.
        # Size of output['hidden_states']: 33

        # if self.aggregation == "max":
        #     embeddings = [
        #         self.model(
        #             self.tokenizer(text, return_tensors="pt")["input_ids"].to(
        #                 self.device
        #             )
        #         )["hidden_states"][self.hidden_state_id]
        #         .max(axis=1)
        #         .values[0]
        #         .cpu()
        #         .detach()
        #         .numpy()
        #         for text in texts
        #     ]
        embeddings = []
        if self.aggregation == "mean":
            for text in texts:
                output = self.model(
                    self.tokenizer(text, return_tensors="pt")["input_ids"].to(
                        self.device
                    )
                )["hidden_states"][self.hidden_state_id]

                # Don't include last token since it represent
                # next token prediction
                output = output[0, :, :]
                output = output.mean(axis=0).cpu().detach().numpy()
                embeddings.append(output)
        elif self.aggregation == "token_embeddings":
            for text in texts:
                output = self.model.model.embed_tokens(
                    self.tokenizer(text, return_tensors="pt")["input_ids"].to(
                        self.device
                    )
                )
                output = output[0, :, :]
                output = output.mean(axis=0).cpu().detach().numpy()
                embeddings.append(output)
        elif self.aggregation == "last_token":
            for text in texts:
                text = f"This sentence: {text} means in one word:"
                output = self.model(
                    self.tokenizer(text, return_tensors="pt")["input_ids"].to(
                        self.device
                    )
                )["hidden_states"][-1]

                # take the last token
                output = output[0, -1, :].cpu().detach().numpy()
                embeddings.append(output)
        else:
            raise NotImplementedError

        logger.debug(
            f"Embedding shape: {embeddings[0].shape}, Total embeddings: {len(embeddings)}"
        )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a LLM model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """

        return self.embed_documents([text])[0]
