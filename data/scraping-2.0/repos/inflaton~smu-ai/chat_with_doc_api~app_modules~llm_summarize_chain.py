import os
from typing import List, Optional
from langchain import PromptTemplate

from langchain.chains.base import Chain
from langchain.chains.summarize import load_summarize_chain

from app_modules.llm_inference import LLMInference


def get_llama_2_prompt_template(instruction):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. Read the text to get context"

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


class SummarizeChain(LLMInference):
    def __init__(self, llm_loader):
        super().__init__(llm_loader)

    def create_chain(self, inputs) -> Chain:
        use_llama_2_prompt_template = (
            os.environ.get("USE_LLAMA_2_PROMPT_TEMPLATE") == "true"
        )
        prompt_template = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""

        if use_llama_2_prompt_template:
            prompt_template = get_llama_2_prompt_template(prompt_template)
        prompt = PromptTemplate.from_template(prompt_template)

        refine_template = (
            "Your job is to produce a final summary\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary."
            "If the context isn't useful, return the original summary."
        )

        if use_llama_2_prompt_template:
            refine_template = get_llama_2_prompt_template(refine_template)
        refine_prompt = PromptTemplate.from_template(refine_template)

        chain = load_summarize_chain(
            llm=self.llm_loader.llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )
        return chain

    def run_chain(self, chain, inputs, callbacks: Optional[List] = []):
        result = chain(inputs, return_only_outputs=True)
        return result
