import json
from llm_rs.langchain import RustformersLLM
from llm_rs import SessionConfig, GenerationConfig
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pathlib import Path
import urllib.request

class ModelDownloader:
    def download_file(self, url, file_path):
        print(f"Downloading {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download completed.")

class Chainer:
    def __init__(self):
        self.model_download = ModelDownloader()
        with open('config.json') as self.configuration:
            self.user_config = json.load(self.configuration)
        self.model = model = "2midguifSfFt5SbHJsxP.bin"

        if not Path(model).is_file():
            self.model_download.download_file(f"", model)


        self.stop_words = ['<EOL>','<eol>', '<Eol>','pertanyaan :','Human', 'human', 'Pertanyaan','\n' ]
        session_config = SessionConfig(
            threads=self.user_config['threads'],
            context_length=self.user_config['context_length'],
            prefer_mmap=False
        )

        generation_config = GenerationConfig(
            top_p=self.user_config['top_p'],
            top_k=self.user_config['top_k'],
            temperature=self.user_config['temperature'],
            max_new_tokens=self.user_config['max_new_tokens'],
            repetition_penalty=self.user_config['repetition_penalty'],
            stop_words=self.stop_words
        )

        template = self.user_config['template']

        self.template = template
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "instruction"],
            template=self.template
        )

        self.memory = ConversationBufferWindowMemory(k=2,memory_key="chat_history",human_prefix='Pertanyaan saya ', ai_prefix='Jawaban anda ')

        self.llm = RustformersLLM(
            model_path_or_repo_id=self.model,
            session_config=session_config,
            generation_config=generation_config,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        self.chain = ConversationChain(input_key='instruction',
                                       llm=self.llm, 
                                       prompt=self.prompt, 
                                       memory=self.memory,
                                       verbose=True)
