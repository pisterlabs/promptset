# Load model directly
import torch
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    GenerationConfig
)

model_id = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=device)
gen_config = GenerationConfig.from_pretrained(model_id)


class lamini:
    def __init__(self) -> None:
        pass

    def load_model(
        task="text2text-generation",
        **kwargs
    ):
        """Returns a pipeline for the model
        - model: MBZUAI/LaMini-Flan-T5-248M

        Returns:
            _type_: _description_
        """

        max_length = kwargs.get("max_length", 512)
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 0.95)
        repetition_penalty = kwargs.get("repetition_penalty", 1.15)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=gen_config,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        return llm


class templates:
    """
    A class that contains methods for generating summaries and titles for text.

    Attributes:
    summaryPipe (None): A pipeline for generating summaries.
    TitlePipe (None): A pipeline for generating titles.
    """

    def __init__(self) -> None:
        self.summaryPipe = None
        self.TitlePipe = None

    def ChunkSummarizer(self, text, custom_instruction: str = None, **kwargs):
        """
        Generates a summary for the given text.

        Args:
        text (str): The text to be summarized.
        custom_instruction (str, optional): Custom instructions for the summarizer. Defaults to None.
        **kwargs: Additional keyword arguments for the summarizer.

        Returns:
        str: The generated summary.
        """

        default_instruction = "summarize for better understanding: "
        instructions = custom_instruction if custom_instruction is not None else default_instruction
        pipe = self.summaryPipe

        max_length = kwargs.get("max_length", 400)
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 0.95)
        repetition_penalty = kwargs.get("repetition_penalty", 1.15)

        if pipe is None:
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                generation_config=gen_config,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        return pipe(instructions + text)[0]["generated_text"]

    def ChunkTitle(self, text, custom_instruction: str = None, **kwargs):
        """
        Generates a title for the given text.

        Args:
        text (str): The text for which a title is to be generated.
        custom_instruction (str, optional): Custom instructions for the title generator. Defaults to None.
        **kwargs: Additional keyword arguments for the title generator.

        Returns:
        str: The generated title.
        """

        default_instruction = "generate a perfect title for the following text in 6 words: "
        instructions = custom_instruction if custom_instruction is not None else default_instruction
        pipe = self.summaryPipe

        max_length = kwargs.get("max_length", 60)
        temperature = kwargs.get("temperature", 0)
        top_p = kwargs.get("top_p", 0.95)
        repetition_penalty = kwargs.get("repetition_penalty", 1.15)

        if pipe is None:
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=max_length,
                generation_config=gen_config,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        return pipe(instructions + text)[0]["generated_text"]

    @staticmethod
    def model():
        """
        Loads the lamini model.

        Returns:
        lamini: The loaded lamini model.
        """
        m = lamini()
        return m.load_model()
