import argparse
from dotenv import load_dotenv
from book import Book,Chapter
from mdbook import MdBook
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from tool import load_file
import log 
from prompt import LLMBasePrompt
from config import BuildConfig

load_dotenv()

# Define LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",timeout=120)

logger = log.setup_logger()


class LLMBookGen:

    def __init__(self,prompt:LLMBasePrompt,conf,llm) -> None:
        """LLMBookGen generates a book based on the prompt."""
        self.book_prompt=prompt
        self.llm=llm 
        self.conf=conf 
    
    
    def gen_book(self,input:str,language:str="english",verbose=True):
        
        title_prompt_tpl=self.book_prompt.title_prompt
        title_prompt=PromptTemplate(template=title_prompt_tpl,input_variables=["input","language"])
        title_chain=LLMChain(llm=llm,prompt=title_prompt,output_key="title",verbose=verbose)

        summary_prompt_tpl=self.book_prompt.summary_prompt
        summary_prompt=PromptTemplate(template=summary_prompt_tpl,input_variables=["title","language"])
        summary_chain=LLMChain(llm=llm,prompt=summary_prompt,output_key="summary",verbose=verbose)

        chapter_prompt_tpl=self.book_prompt.toc_prompt
        chapter_prompt=PromptTemplate(template=chapter_prompt_tpl,input_variables=["summary","language"])
        chapter_chain=LLMChain(llm=llm,prompt=chapter_prompt,output_key="chapters",verbose=verbose)

        # core chain 
        core_chain=SequentialChain(chains=[title_chain,summary_chain,chapter_chain],input_variables=["input","language"], output_variables=["title","summary","chapters","language"])
        result=core_chain({"input":input,"language":language})

        return self.as_book(result) 
    

    def as_book(self,result,author="GPT") ->Book:
        chapters=result['chapters']
        chapterslines=chapters.split("\n")
        title=result['title']
        summary=result['summary']
        language=result['language']

        book=Book(title=title,summary=summary,author=author)
        
        for c in chapterslines:
            seqc=c.split("::::")
            if len(seqc)>=3:
                nu=int(seqc[0])
                title=seqc[1]
                summary=seqc[2]
                # Content requires the next GPT session.
                the_chapter=Chapter(nu=nu,title=title,summary=summary)
                book.add_chapter(self.gen_chapter(the_chapter,book.title,book.summary,language=language))

        return book

    def gen_chapter(self,chapter:Chapter,book_title:str,book_summary:str,language:str):
        promot_tpl=self.book_prompt.content_detail_prompt
        prompt=PromptTemplate(template=promot_tpl,input_variables=["title","summary","subtitle","subsummary","language"])

        title_chain=LLMChain(llm=llm,prompt=prompt,output_key="content",verbose=True)
        result=title_chain({"title":book_title,"summary":book_summary,"subtitle":chapter.title,"subsummary":chapter.summary,"language":language})
        if "content" in result:
            chapter.content=self._parse_content(result["content"])
        else:
            chapter.content="**** FAIL GEN {} **** ".format(chapter)

        return chapter

    def _parse_content(self,content:str):

        prefix="Result:"
        if content.startswith(prefix):
            return content[len(prefix):]
        return content 

def main():
    parser = argparse.ArgumentParser(description="Generate and build a book using LLM and Markdown.")
    parser.add_argument("--input", type=str, help="The input text for the book.")
    parser.add_argument("--location", type=str, help="The location to save the generated book.")
    parser.add_argument("--language", type=str, help="The language of the generated book.",default="english")
    parser.add_argument("--authors", type=str, help="The author of the generated book.", default="Unknown")
    parser.add_argument("--config", type=str, help="Specify the name of the config file.", default="buildbook.json")

    args = parser.parse_args()

    if not args.input or not args.location:
        print("Both --input and --location are required.")
        return
    
    conf=BuildConfig.from_file(args.config)
    llm_book_gen = LLMBookGen(llm=llm, prompt=LLMBasePrompt().load(conf=conf),conf=conf)
    book = llm_book_gen.gen_book(args.input,language=args.language)
    book.authors= str(args.authors).split(",") 
    MdBook(book, args.location).build()

if __name__=="__main__":
    main()