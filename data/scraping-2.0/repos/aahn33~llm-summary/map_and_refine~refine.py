from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks import get_openai_callback
import tiktoken
import os

class TextRefiner:
    def __init__(self, llm, model_name):
        self.llm = llm
        self.model_name = model_name
        self.total_tokens = 0

        # Setup summarization and refinement prompts
        sum_prompt_template = """
                      Write a summary of the text that includes the main points and any important details in paragraph form.
                      {text}
                      """
        self.sum_prompt = PromptTemplate(template=sum_prompt_template, input_variables=["text"])

        refine_prompt_template = '''
        Your assignment is to expand an existing summary by adding new information that follows it. Here's the current summary up to a specified point:

        {existing}

        Now, consider the following content which occurs after the existing summary:

        {text}

        Evaluate the additional content for its relevance and importance in relation to the existing summary. If this new information is significant and directly relates to what has already been summarized, integrate it smoothly into the existing summary to create a comprehensive and cohesive final version. If the additional content doesn't provide substantial value or isn't relevant to the existing summary, simply return the original summary as it is. If the summary is getting too long you can shorten it by removing unnecessary details.

        Your final output must only be the comprehensive and cohesive final version of the summary. It should contain no other text, such as reasoning behind the summary.

        Summary:
                '''
        self.refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["existing", "text"])

        # Load chains
        self.sum_chain = load_summarize_chain(llm, chain_type="stuff", prompt=self.sum_prompt)
        self.refine_chain = load_summarize_chain(llm, chain_type="stuff", prompt=self.refine_prompt)

        # Setup text splitter
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3500, chunk_overlap=0)

    def refine(self, text):
        texts = self.text_splitter.split_text(text)
        print(f"The text was split into {len(texts)} chunks.")
        texts_docs = [[Document(page_content=text)] for text in texts]

        cur_summary = ""
        for num, chunk in enumerate(texts_docs):
            with get_openai_callback() as cb:
                print(f"Processing chunk {num}")
                chunk_sum = self.sum_chain.run(chunk)
                self.save_to_file(chunk_sum, "chunk_sum", num)
                input = {'existing': cur_summary, 'input_documents': [Document(page_content=chunk_sum)]}

                cur_summary = self.refine_chain.run(input)
                self.total_tokens += cb.total_tokens

            self.save_to_file(cur_summary, "cur_summary", num)

        return cur_summary, self.total_tokens

    def save_to_file(self, text, name, iteration):
        directory = "refine"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f"{name}_{iteration}.txt"
        with open(os.path.join(directory, filename), 'w', encoding='utf-8') as file:
            file.write(text)

if __name__ == '__main__':
    # Usage of the TextRefiner class
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0, openai_api_key="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", model_name=model_name)
    refiner = TextRefiner(llm, model_name)

    file_path = 'Gatsby.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        book_text = file.read()

    summary, total_tokens = refiner.refine(book_text)
    print(f"Total tokens: {total_tokens}")
    print(summary)
