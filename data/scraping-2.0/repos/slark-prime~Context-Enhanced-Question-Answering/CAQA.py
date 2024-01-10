import logging
import torch
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)
from peft import PeftModel, PeftConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
FROM_HUB = False


class CAQA:
    """Initializes the CAQA class with default values."""
    def __init__(self):
        self.fine_tuned = False
        self.peft_model_name = None
        self.language_model_name = "lmsys/vicuna-7b-v1.3"
        self.embedding_model = "hkunlp/instructor-xl"

        self.qa_chain = None
        self.core_model = None

        # model params that make difference to model generations
        self.llm_kwargs = {
            "temperature": 0.001,
            # "max_new_tokens": 500,
            "max_length": 2048,
            "repetition_penalty": 1.2,
            # "repetition_penalty": 1.7, # for falcon
            # "repetition_penalty": 1.2,
            "top_k": 60,
            "top_p": 0.92,
            "device": "cuda:0"
        }

    @torch.inference_mode()
    def generate_response(self, query):
        """
        Generates a response for the given query.

        :param query: The question to be answered.
        :return: The answer and source documents.
        """
        with get_openai_callback() as cb:  # calculate token usage if openai LLM is used
            # build the qa chain
            response = self.qa_chain(query)
            answer, source_docs = response['result'], response['source_documents']

            if self.language_model_name == "openai":
                print(cb)

        return answer, source_docs


class CAQABuilder:
    def __init__(self):
        """Initializes the CAQABuilder class with default values."""

        self.caqa = CAQA()
        self.chain_type = "stuff"
        self.task = "text-generation"

        self.prompt_template = """For the task ahead, you should generate a precise answer to the question below. Use 
        the context provided to identify relevant information. The context may contain both relevant and irrelevant 
        details, so focus on what is directly related to the question. Your answer should be specific, concise, 
        and accurate. If the context doesn't provide sufficient information to answer the question, just say that you 
        do not know the answer. """

    def set_embedding_model(self, embedding_model):
        """Sets the embedding model for the CAQA instance."""
        self.caqa.embedding_model = embedding_model
        return self

    def set_llm(self, llm_repo_id):
        """Sets the language model repository ID for the CAQA instance."""
        return self

    def set_llm_params(self, **kwargs):
        """Sets the language model parameters for the CAQA instance."""
        self.caqa.llm_kwargs.update(kwargs)
        return self

    def set_chain_type(self, chain_type: str):
        """
        Sets the chain type for the CAQA instance.

        :param chain_type: The chain type. Options include "stuff", "map_reduce", "map_rerank", and "refine".
        """
        self.chain_type = chain_type
        return self

    def set_prompt(self, prompt):
        """Sets the prompt for the CAQA instance."""
        self.prompt_template = prompt

        return self

    def set_peft_model(self, model_name):
        """Sets the PEFT model for the CAQA instance."""
        self.caqa.fine_tuned = True
        self.caqa.peft_model_name = model_name

        return self

    def load_embedding_model(self):
        """
        Loads the embedding model for the CAQA instance.

        :return: The loaded embedding model.
        """
        # Embedding Model
        embedding_model = self.caqa.embedding_model
        if "hkunlp/instructor" in embedding_model:
            return HuggingFaceInstructEmbeddings(model_name=embedding_model, model_kwargs={"device": "cuda"})
        elif embedding_model == "openai":
            return OpenAIEmbeddings()
        else:
            return HuggingFaceEmbeddings(model_name=embedding_model)

    def load_vectorstore(self):
        """
        Loads the vectorstore for the CAQA instance.

        :return: The loaded vectorstore.
        """
        # Load the vectorstore
        db = Chroma(persist_directory=PERSIST_DIRECTORY,
                    embedding_function=self.load_embedding_model(),
                    client_settings=CHROMA_SETTINGS)

        return db.as_retriever()

    def load_peft_config(self):
        """Loads the PEFT config for the CAQA instance."""
        peft_model_id = self.caqa.peft_model_name
        self.caqa.language_model_name = PeftConfig.from_pretrained(peft_model_id).base_model_name_or_path
        logging.info("PEFT config loaded")

    def load_model(self):
        """Loads the model for the CAQA instance."""
        if self.caqa.language_model_name == "openai":
            self.caqa.core_model = OpenAI()

        if self.caqa.fine_tuned:
            self.load_peft_config()
            model = PeftModel.from_pretrained(self._load_model(), self.caqa.peft_model_name)
            logging.info("Peft model loaded")

        else:
            model = self._load_model()

        model_name = self.caqa.language_model_name

        logging.info("Loading Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load configuration from the model to avoid warnings
        # generation_config = GenerationConfig.from_pretrained(model_name)

        # Create a pipeline for text generation

        self.caqa.core_model = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                # generation_config=generation_config,
                **self.caqa.llm_kwargs
            )
        )

    def _load_model(self):
        """
        Loads the model for the CAQA instance.

        :return: The loaded model.
        """
        model = None

        model_name = self.caqa.language_model_name
        logging.info(f"Loading Model: {model_name}")

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         device_map="auto",
                                                         torch_dtype=torch.float16,
                                                         # low_cpu_mem_usage=True,
                                                         trust_remote_code=True,
                                                         # max_memory={0: "7GB"} # change according to RAM available
                                                         )
            model.tie_weights()

        except Exception as e:
            logging.error("Error loading the language model: %s", str(e))

        logging.info("Local LLM Loaded")

        return model

    def build_qa_chain(self):

        retriever = self.load_vectorstore()

        # Load the language model
        self.load_model()

        chain_type_kwargs = {"prompt": self.build_prompt_template()}
        return RetrievalQA.from_chain_type(
            llm=self.caqa.core_model,
            chain_type=self.chain_type,
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

    def build(self):

        self.caqa.qa_chain = self.build_qa_chain()

        return self.caqa

    def build_prompt_template(self):
        from langchain.prompts import PromptTemplate
        prompt_template = self.prompt_template + """Now use the following pieces of context to answer the question at the end.


                    {context}

                    Question: {question}
                    Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        return PROMPT
