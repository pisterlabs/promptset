from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks import get_openai_callback
from concurrent.futures import ThreadPoolExecutor, as_completed
from refine import TextRefiner
import tiktoken
import os

# TODO: Import this from other file, I was getting an error when I tried to import it
# TODO: Figure out why this method uses 5x as many tokens as just refine


class MapRefineTextSummarizer:
    def __init__(self, llm, model_name, refine_size = 4):
        self.llm = llm
        self.recursive_calls = 0
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.total_tokens_used = 0
        self.refiner = TextRefiner(llm, model_name)
        self.refine_size = refine_size

        # Setup prompts
        self.map_prompt = PromptTemplate(
            template="""Write a summary of this text without listing. {text}""",
            input_variables=["text"]
        )

        self.combine_prompt = PromptTemplate(
            # template="""Can you create a comprehensive summary from these mini-summaries. Your output should be a couple paragraphs long. Only use the text provided to generate the summary. 
            #             {text}
            #          """, # For some reason, this prompt makes the model generate a summary unrelated to the input, it recognizes the book
            template = """Write a summary of this text without listing
                           {text}""",
            input_variables=["text"]
        )

        # Load chains
        self.map_chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=self.map_prompt)
        self.reduce_chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=self.combine_prompt)

    def summarize_chunk(self, chunk):
        """ Function to summarize a single chunk of text """
        # with get_openai_callback() as cb:
        #     summary = self.map_chain.run([Document(page_content=chunk)])
        #     self.total_tokens_used += cb.total_tokens
        summary, tokens = self.refiner.refine(chunk)
        print(f"Tokens used in refine step: {tokens}")
        self.total_tokens_used += tokens
        return summary
    
    def summarize_text(self, text):
        # Split text into chunks
        # Important: The chunk size is now a multiple of the refine_size
        chunk_size = 3500 * self.refine_size
        print(f"Chunk size: {chunk_size}")
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=0, separator='\n'
        )
        texts = text_splitter.split_text(text)
        self.save_to_file("\n-------------------------------------------------------\n".join(texts), "chunks",self.recursive_calls)
        print("Map: The text was split into " + str(len(texts)) + " chunks")

        summaries = [None] * len(texts)  # Initialize a list of the correct size with placeholders
        # Use ThreadPoolExecutor to process chunks in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all chunks to be processed
            future_to_chunk = {executor.submit(self.summarize_chunk, chunk): i for i, chunk in enumerate(texts)}
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]  # Get the index of the chunk
                print(f"Processed chunk {chunk_index}")
                summaries[chunk_index] = future.result()  # Place summary in the corresponding index

        # Filter out any placeholders in case some tasks failed
        summaries = [summary for summary in summaries if summary is not None]

        # Concatenate all summaries into a single string, preserving the order
        return "\n\n".join(summaries)


    def reduce_summaries(self, summaries):
        with get_openai_callback() as cb:
            final_summary = self.reduce_chain.run([Document(page_content=summaries)])
            self.total_tokens_used += cb.total_tokens
        return final_summary

    def process(self, text):
        to_be_summarized = text
        iteration = 0
        while len(self.encoding.encode(to_be_summarized)) > 4000:
            print("Performing the map step")
            # Before summarizing, save the current state of 'to_be_summarized' to a file
            self.save_to_file(to_be_summarized, "summary",iteration)
            to_be_summarized = self.summarize_text(to_be_summarized)
            self.recursive_calls += 1
            iteration += 1
        # print(f"Total tokens to summarize from map step: {len(self.encoding.encode(to_be_summarized))}")
        self.save_to_file(to_be_summarized,"summary", iteration)
        final_summary = self.reduce_summaries(to_be_summarized)

        self.save_to_file(final_summary,"final_summary", iteration)
        return final_summary, self.total_tokens_used
    
  
    def save_to_file(self, text, name, iteration):
        # Make sure to replace 'path_to_directory' with the actual path where you want to save the files
        directory = "map-refine"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f"{name}_{iteration}.txt"
        with open(os.path.join(directory, filename), 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Saved iteration {iteration} to file")

model_name = "gpt-3.5-turbo"

# Create an instance of the ChatOpenAI
llm = ChatOpenAI(
    temperature=0,
    openai_api_key="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",
    model_name=model_name
)

if __name__ == '__main__':
    # Initialize the TextSummarizer with the llm instance
    summarizer = MapRefineTextSummarizer(llm=llm, model_name=model_name, refine_size=4)

    # The text to be summarized is passed here
    file_path = 'Gatsby.txt'

    # Open the text file and read its content into a string variable
    with open(file_path, 'r', encoding='utf-8') as file:
        book_text = file.read()

    # Call the process method with the text
    summary, total_tokens_used = summarizer.process(book_text)
    print(summary)
    print(f"Total tokens used: {total_tokens_used}")
