import os
from .ApiOpenAi import ApiOpenAi

from templates import *
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class ApiChatGpt(ApiOpenAi):
    def __init__(self,dry_run=False):
        super().__init__(dry_run=dry_run)
        self.provider_name="ChatGPT"
        self.dry_run_object=dry_run_story
        self.recyclable_item=""
        self.temperature=0.9
        self.as_string=True
        self.language="English"
        

    def set_language(self,language):
        self.language=language

    def fetch_prompt(self,item="metal can",prompt_template='pollution_prompt_template',**kwargs):
        llm = OpenAI(model_name="gpt-4",temperature=self.temperature,openai_api_key=self.api_key)
        # prompt = PromptTemplate(
        #     input_variables=["item"],
        #     template=eval(prompt_template),
        # )

        if not self.dry_run:
            prompt=eval(prompt_template)+f" {item} | {self.language}"
            story=llm(prompt)
            if self.as_string:
                return story

            filename=f"{item}_{story[:20].replace(' ','')}"
            if len(filename)>250:
                filename=filename[:250]

            story_folder="pollution_scenes"
            with open(f"{story_folder}/{filename}.txt", "w") as file:
                # Write the string to the file
                file.write(story)

            print(f"pollution prompt saved at {story_folder}/{filename}.txt")
            return story
        
        else:
            return eval("dry_run_"+prompt_template)

    def translate(self,word,language=None):
        if not language:
            language=self.language="English"
        llm = OpenAI(model_name="gpt-4",temperature=0.1,openai_api_key=self.api_key)
        translated_word=llm(f"what is '{word}' in {language}. Just return the answer")
        return translated_word
    

    def fetch_using_index(self,query = "soda can",language=None):
        """
        call self.produce_index_pinecone(self,index_name='cfc',docs_dir="recycling_data_dir",embeddings=None) first
        """
        if not language:
            language=self.language
        if not self.dry_run:
            self.produce_index_pinecone(index_name='cfc',docs_dir="recycling_data")

            qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.5), chain_type="stuff", return_source_documents=True ,retriever=self.index.as_retriever(search_type='mmr',search_kwargs={"k":1}))

            response = qa({"query": query}, return_only_outputs=True)

            if "I don't know" in response['result']:
                return {'result':None,'source':None}
            
            result=response['result']
            source=response['source_documents'][0].metadata['source']

            return {'result':result,'source':source}

        else:
            return dry_run_index_fetch

            
