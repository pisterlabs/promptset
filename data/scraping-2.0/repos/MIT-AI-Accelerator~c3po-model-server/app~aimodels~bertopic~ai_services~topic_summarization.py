import os
from pathlib import Path
from string import punctuation
from langchain import PromptTemplate
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.llamacpp import LlamaCpp
from app.core.logging import logger
from app.core.model_cache import MODEL_CACHE_BASEDIR
from app.core.minio import download_file_from_minio
from app.core.config import get_acronym_dictionary
from app.aimodels.gpt4all.models.gpt4all_pretrained import Gpt4AllPretrainedModel, Gpt4AllModelFilenameEnum
from app.aimodels.gpt4all.crud import crud_gpt4all_pretrained as crud

# default templates for topic summarization
MAP_PROMPT_TEMPLATE = """
Write a summary of the following text that includes the main points and important details.
{text}
"""
COMBINE_PROMPT_TEMPLATE = """
Write a concise summary in 1-3 sentences which covers the key points of the text.
{text}
TEXT SUMMARY:
"""


# default parameters for topic summarization
DEFAULT_N_REPR_DOCS = 5
DEFAULT_LLM_TEMP = 0.8
DEFAULT_LLM_TOP_P = 0.95
DEFAULT_LLM_REPEAT_PENALTY = 1.3


class TopicSummarizer:

    def __init__(self):
        self.model_type = None
        self.model_id = None
        self.map_prompt_template = None
        self.combine_prompt_template = None
        self.temp = None
        self.top_p = None
        self.top_p = None
        self.llm = None

    def initialize_llm(self, s3, model_obj,
                       map_prompt_template=MAP_PROMPT_TEMPLATE,
                       combine_prompt_template=COMBINE_PROMPT_TEMPLATE,
                       temp=DEFAULT_LLM_TEMP,
                       top_p=DEFAULT_LLM_TOP_P,
                       repeat_penalty=DEFAULT_LLM_REPEAT_PENALTY):

        self.model_type = model_obj.model_type
        self.model_id = model_obj.id
        llm_path = os.path.join(MODEL_CACHE_BASEDIR, self.model_type)

        # download gpt4all model binary
        if not os.path.isfile(llm_path):
            # Create the directory if it doesn't exist
            Path(llm_path).parent.mkdir(parents=True, exist_ok=True)

            # Download the file from Minio
            logger.info(f"Downloading model from Minio to {llm_path}")
            download_file_from_minio(model_obj.id, s3, filename=llm_path)

            if not os.path.isfile(llm_path):
                logger.error(f"Error downloading model from Minio to {llm_path}")
            else:
                logger.info(f"Downloaded model from Minio to {llm_path}")

        # initialize llm
        self.llm = GPT4All(
            model=llm_path,
            callbacks=[StreamingStdOutCallbackHandler()],
            verbose=False,
            n_predict=256,
            temp=temp,
            top_p=top_p,
            echo=False,
            stop=[],
            repeat_penalty=repeat_penalty,
            use_mlock=False
        )
        self.map_prompt_template = map_prompt_template
        self.combine_prompt_template = combine_prompt_template

        # TODO add configuration parameters for temp, top_p, and repeat_penalty
        # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=36312850
        self.temp = temp
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

    # check existing llm
    def check_parameters(self, model_id, map_prompt_template, combine_prompt_template):
        return self.model_id == model_id and self.map_prompt_template == map_prompt_template and self.combine_prompt_template == combine_prompt_template

    # TODO add configuration parameters for temp, top_p, and repeat_penalty
    # https://github.com/orgs/MIT-AI-Accelerator/projects/2/views/1?pane=issue&itemId=36312850
    # def check_parameters(self, map_prompt_template, combine_prompt_template, temp, top_p, repeat_penalty):
    #     return self.map_prompt_template == map_prompt_template & self.combine_prompt_template == combine_prompt_template & self.temp == temp & self.top_p == top_p & self.repeat_penalty == repeat_penalty

    # Replaces acronyms in text with expanded meaning from dictionary
    def replace_acronyms(self, d, text):
        return ' '.join(d[x.upper()] if x.upper() in d else x for x in text.split())

    # Fixes text after preprocessing by adding back punctuation and replacing acronyms
    def fix_text(self, docs):
        acronym_dictionary = get_acronym_dictionary()
        fixed_docs = []
        for text in docs:
            if text.endswith('.') or (text != "" and not text.endswith('.')):
                fixed_docs.append(self.replace_acronyms(
                    acronym_dictionary, text.rstrip(punctuation)) + '.')
            elif text.endswith('?'):
                fixed_docs.append(self.replace_acronyms(
                    acronym_dictionary, text.rstrip(punctuation)) + '?')
            elif text.endswith('!'):
                fixed_docs.append(self.replace_acronyms(
                    acronym_dictionary, text.rstrip(punctuation)) + '!')
            else:
                fixed_docs.append(text)
        return fixed_docs

    # Function to summarize list of texts using LangChain map-reduce chain with custom prompts.
    def get_summary(self, documents):
        if self.llm is None:
            logger.error("TopicSummarizer not initialized")
            return None

        # replace acronyms and concatenate top n documents
        list_of_texts = '\n'.join(self.fix_text(documents))
        num_tokens = self.llm.get_num_tokens(list_of_texts)
        if num_tokens > self.llm.max_tokens:
            logger.error("Skipping summarization, exceeded max tokens: %d" % num_tokens)
            return None

        text_splitter = CharacterTextSplitter()
        # stuffs the lists of text into "Document" objects for LangChain
        docs = text_splitter.create_documents([list_of_texts])

        map_prompt = PromptTemplate(
            template=self.map_prompt_template, input_variables=["text"])

        combine_prompt = PromptTemplate(
            template=self.combine_prompt_template, input_variables=["text"]
        )

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        chain = load_summarize_chain(self.llm,
                                     chain_type="map_reduce",
                                     verbose=False,
                                     map_prompt=map_prompt,
                                     combine_prompt=combine_prompt,
                                     return_intermediate_steps=True,
                                     input_key="input_documents",
                                     output_key="output_text",
                                     )

        return chain({"input_documents": docs})


topic_summarizer = TopicSummarizer()
