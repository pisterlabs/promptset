import os
from loguru import logger
from sprite_ai.constants import APP_NAME
from sprite_ai.language.language_model import LanguageModel
from sprite_ai.language.language_model_config import LanguageModelConfig
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory, ConversationSummaryMemory
from sprite_ai.language.llm_factory import LLMFactory

class LanguageModelFactory:
    def _build_prompt_template(self, model_config: LanguageModelConfig) -> PromptTemplate:
        prompt_template = model_config.prompt_template
        prompt_template = prompt_template.format(
            system_prompt=model_config.system_prompt,
            chat_history="{chat_history}",
            user_input="{user_input}",
        )
        logger.debug(prompt_template)

        prompt = PromptTemplate(
            input_variables=["chat_history", "user_input"], template=prompt_template
        )
        return prompt
    
    def _build_memory(self, model_config: LanguageModelConfig, llm: LLM):        
        if model_config.memory_type == 'summary':
            Memory = ConversationSummaryMemory
        elif model_config.memory_type == 'summary_buffer':
            Memory = ConversationSummaryBufferMemory
        else:
            raise ValueError(f'Unsuported memory type: {model_config.memory_type}')

        memory = Memory(
            llm=llm,
            memory_key="chat_history",
            human_prefix=model_config.user_prefix,
            ai_prefix=model_config.ai_prefix,
            max_token_limit=model_config.memory_tokens_limit,
        )
        return memory
    
    def _build_llm(self, model_config: LanguageModelConfig):
        llm_factory = LLMFactory()
        llm = llm_factory.build(
            model_config.name,
            model_config.context_size,
            model_config.model_temperature,
            model_config.url,
            model_config.stop_strings,
            model_config.api_key,
        )
        
        return llm
    
    def _build_llm_chain(self, model_config : LanguageModelConfig) -> LLMChain:
        llm = self._build_llm(model_config)
        prompt_template = self._build_prompt_template(model_config)
        memory = self._build_memory(model_config, llm)

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            memory=memory,
            verbose=False,
        )
        return llm_chain
    
    def build(self, model_config : LanguageModelConfig) -> LanguageModel:
        llm_chain = self._build_llm_chain(model_config)
        language_model = LanguageModel(llm_chain)

        return language_model