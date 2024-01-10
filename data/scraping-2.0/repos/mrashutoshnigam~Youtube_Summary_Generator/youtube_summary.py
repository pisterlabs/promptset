from langchain.document_loaders import YoutubeLoader
from getpass import getpass
import os
from langchain.llms import Replicate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate


class Youtube:
    def __init__(self):
        self.docs = None

    def load_youtube_video(self, url):
        loader = YoutubeLoader.from_youtube_url(
            url
        )
        self.docs = loader.load()

    def configure_llama(self):
        # enter your Replicate API token, or you can use local Llama. See README for more info
        REPLICATE_API_TOKEN = 'r8_duxtnDKCMHgm3zubWmGycyVGlgPv1TT0BLvf4'
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        llama2_13b = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
        self.llm = Replicate(
            model=llama2_13b,
            model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens": 500}
        )

    def generate_short_summary(self):
        prompt = ChatPromptTemplate.from_template(
            "Give me a summary of the text below: {text}?"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        # be careful of the input text length sent to LLM
        self.text = self.docs[0].page_content[:1000]
        summary = chain.run(self.text)
        # this is the summary of the first 4000 characters of the video content
        return summary

    def entire_summary(self):
        prompt = ChatPromptTemplate.from_template(
            "Give me a summary of the text below: {text}?"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        text = self.docs[0].page_content
        summary = chain.run(text)
        return summary

    def generate_summary(self, youtube_url):
        # self.load_youtube_video(youtube_url)
        videoInfo = self.load_video_info(youtube_url)
        self.configure_llama()
        videoInfo["summary"] = self.entire_summary()
        return videoInfo

    def use_local(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path='model/ggml-model-q4_0.gguf',
            # n_ctx=6000,
            max_tokens=32,
            verbose=True)

    def load_video_info(self, url):
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=True
        )
        docs = loader.load()
        self.docs = docs
        if docs is not None and len(self.docs) > 0:
            video = {
                "title": docs[0].metadata['title'],
                "thumbnail_url": docs[0].metadata['thumbnail_url'],
                "publish_date": docs[0].metadata['publish_date'],
                "author": docs[0].metadata['author'],
                "page_content": docs[0].page_content
            }
        else:
            video = None
        return video
