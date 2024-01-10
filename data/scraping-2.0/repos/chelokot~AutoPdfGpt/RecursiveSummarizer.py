import copy
from OpenAIManager import OpenAIManager


class RecursiveSummarizer:
    def __init__(self, openAI_key):
        self.open_ai_manager = OpenAIManager(openAI_key)

    def summarize_chunk(self, chunk):
        return self.open_ai_manager.get_answer(f'"{chunk}"\n Summarize this into {len(chunk.split(" "))//2} words.', chunk)
    
    def get_with_summarized_chunks(self, chunks):
        if(len(chunks) < 2):
            return chunks
        summarized_chunks = []
        for i in range(0, len(chunks), 2):
            chunk1 = chunks[i]
            chunk2 = chunks[i+1]
            summarized_chunk1 = self.summarize_chunk(chunk1)
            summarized_chunk2 = self.summarize_chunk(chunk2)
            summarized_chunk  = summarized_chunk1 + ' ' + summarized_chunk2
            summarized_chunks.append(summarized_chunk)
        return chunks + self.add_summarized_chunks(summarized_chunks)

             