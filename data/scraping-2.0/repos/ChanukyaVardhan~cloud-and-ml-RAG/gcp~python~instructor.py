from enum import Enum
from InstructorEmbedding import INSTRUCTOR
from langchain.text_splitter import RecursiveCharacterTextSplitter

import numpy as np
import torch


class InstructorModelType(Enum):
    BASE = 'hkunlp/instructor-base'
    LARGE = 'hkunlp/instructor-large'
    XL = 'hkunlp/instructor-xl'

class Instructor:

    def __init__(self, instructor_model_type: str, instruction: str, device: str = 'cpu'):
        self.device = device
        self.model = INSTRUCTOR(instructor_model_type).to(self.device)
        self.instruction = instruction

        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=[". ", "\n", " ", ""],
            chunk_size=1800,
            chunk_overlap=360, # 20% overlap
            length_function=len,
        )
        self.chunk_batch_size = 5

    def get_embedding(self, text, split_chunks = True):
        if split_chunks:
            chunks = self.text_splitter.split_text(text)

            all_embeddings = []
            for i in range(0, len(chunks), self.chunk_batch_size):
                batch = chunks[i:i + self.chunk_batch_size]
                embeddings = self._get_batch_embedding(batch, self.instruction)
                all_embeddings.extend(embeddings)

            return chunks, np.array(all_embeddings)

        else:
            with torch.no_grad():
                outputs = self.model.encode([[self.instruction, text]])

            return text, outputs

    def _get_batch_embedding(self, texts, instruction):
        # Pair each text with the common instruction
        text_instruction_pairs = [{"instruction": instruction, "text": text} for text in texts]

        # Prepare texts with instructions for the model
        texts_with_instructions = [[pair["instruction"], pair["text"]] for pair in text_instruction_pairs]
        with torch.no_grad():
            customized_embeddings = self.model.encode(texts_with_instructions)

        return customized_embeddings
