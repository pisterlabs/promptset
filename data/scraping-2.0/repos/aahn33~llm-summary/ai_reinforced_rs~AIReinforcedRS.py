from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import tiktoken
import random

class Reinforced:
    
    def __init__(self, llm, chunk_size, chunk_percentage):
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_percentage = chunk_percentage
        self.total_tokens_used = 0
    
    def split(self, text):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size, chunk_overlap=0)
        texts = text_splitter.split_text(text)
        return texts
    
    def select_percentage(self, texts):
        num_elements = int(len(texts) * (self.chunk_percentage / 100.0)) 
        selected_text = random.sample(list(enumerate(texts)), num_elements)
        return selected_text
    
    def generate_summary(self, selected_text):
        sys_message_summary = SystemMessage(content=(
            "Write a summary of this chunk of text that includes the main points and any important details."
        ))

        summaries = []
        for i, chunk in enumerate(selected_text):
            with get_openai_callback() as cb:
                summaries.append((self.llm([sys_message_summary, HumanMessage(content=chunk[1])]), chunk[0])) #keep chunk index in tuple
                self.total_tokens_used += cb.total_tokens
        return summaries
    
    def incomplete_chunks(self, summaries):
        sys_message_incomplete = SystemMessage(content=(
            "Given these mini summaries with their corresponding ids (in the form \"summary\" id=id), return the ids of" +
            "the mini summaries that don't relate to the rest of the summaries. You only output the ids as integers in the form \"id1,id2,...idn\"."
        ))

        combined_summaries = ''
        for sum in summaries:
            combined_summaries += '(' + str(sum[0]) + ' id=' + str(sum[1]) + '),'

        incomplete = None
        with get_openai_callback() as cb:
            incomplete = self.llm([sys_message_incomplete, HumanMessage(content=combined_summaries)])

        return incomplete
    
    def relevant_chunks_incomplete(self, summaries, incomplete, texts):
        content = incomplete.content
        more_info = [(int(float(num.strip())) if num.strip().isnumeric() else -1)for num in content.split(',')]
        nearby = []
        for sum in summaries:
            nearby.append(sum[1])
        for c in more_info:
            if c > 5:
                if c-5 not in nearby:
                    nearby.append(c-5)
            if c < len(texts)-5:
                if c+5 not in nearby:
                    nearby.append(c+5)
        nearby.sort()
        return nearby
    
    def relevant_chunks(self, nearby, texts):
        sys_message_summary = SystemMessage(content=(
            "Write a summary of this chunk of text that includes the main points and any important details."
        ))
        summaries_final = []
        for i in nearby:
            with get_openai_callback() as cb:
                summaries_final.append(self.llm([sys_message_summary, HumanMessage(content=texts[i])]))
                self.total_tokens_used += cb.total_tokens
        return summaries_final
    
    def final_run(self, text):
        texts = self.split(text)
        selected_text = self.select_percentage(texts)
        summaries = self.generate_summary(selected_text)
        incomplete = self.incomplete_chunks(summaries)
        nearby = self.relevant_chunks_incomplete(summaries, incomplete, texts)
        summaries_final = self.relevant_chunks(nearby, texts)
        return summaries_final, self.total_tokens_used        
      
        
        
if __name__ == "__main__":
    
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key="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",
        model_name="gpt-3.5-turbo"
    )
    
    chunk_size = 35000
    chunk_percentage = 50
    FILE_PATH = 'Gatsby.txt'
    f = open(FILE_PATH, encoding='utf-8')
    text = f.read()
    
    reinforced = Reinforced(llm, chunk_size, chunk_percentage)
    result = reinforced.final_run(text)
    print(result)
    