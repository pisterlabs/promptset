from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import load_prompt

from app.config.settings import get_settings

app_settings = get_settings()


class LlamaStoryTeller:
    def load_llm(self) -> None:
        config = {
            "max_new_tokens": app_settings.STORY_DESIGNER_MAX_TOKENS,
            "gpu_layers": app_settings.STORY_DESIGNER_N_GPU_LAYERS,
            "batch_size": app_settings.STORY_DESIGNER_N_BATCH,
            "repetition_penalty": app_settings.STORY_DESIGNER_REPEAT_PENALTY,
            "temperature": app_settings.STORY_DESIGNER_TEMPERATURE,
            "stop": app_settings.STORY_DESIGNER_STOP,
            "context_length": app_settings.STORY_DESIGNER_N_CTX,
            "stream": True,
        }
        self._llm = CTransformers(
            model=app_settings.STORY_DESIGNER_REPO_ID,
            model_file=app_settings.STORY_DESIGNER_MODEL_FILENAME,
            verbose=app_settings.STORY_DESIGNER_VERBOSE,
            model_type="llama",
            config=config,
            callbacks=[StreamingStdOutCallbackHandler()],
        )  # type: ignore

    def generate_story(self, input_text: str) -> str:
        chain = LLMChain(
            llm=self._llm,
            prompt=load_prompt(app_settings.STORY_DESIGNER_PROMPT_PATH),
            verbose=app_settings.STORY_DESIGNER_VERBOSE,
        )
        return chain.run(input_text)

    def brainstorm(self, input_text: str) -> str:
        chain = LLMChain(
            llm=self._llm,
            prompt=load_prompt(app_settings.STORY_DESIGNER_BRAINSTORMING_PROMPT_PATH),
            verbose=app_settings.STORY_DESIGNER_VERBOSE,
        )
        return chain.run(input_text)

    def choose_idea(self, story_topic: str, ideas: str) -> str:
        chain = LLMChain(
            llm=self._llm,
            prompt=load_prompt(app_settings.STORY_DESIGNER_IDEA_CHOICE_PROMPT_PATH),
            verbose=app_settings.STORY_DESIGNER_VERBOSE,
        )
        return chain.run({"story_topic": story_topic, "ideas": ideas})

    def outline_sequence_of_events(self, story_topic: str, idea: str) -> str:
        chain = LLMChain(
            llm=self._llm,
            prompt=load_prompt(app_settings.STORY_DESIGNER_EVENTS_OUTLINE_PROMPT_PATH),
            verbose=app_settings.STORY_DESIGNER_VERBOSE,
        )
        return chain.run({"story_topic": story_topic, "idea": idea})

    def design_characters(self, story_topic: str, events_outline: str) -> str:
        chain = LLMChain(
            llm=self._llm,
            prompt=load_prompt(app_settings.STORY_DESIGNER_CHARACTERS_DESIGNER_PROMPT_PATH),
            verbose=app_settings.STORY_DESIGNER_VERBOSE,
        )
        return chain.run({"story_topic": story_topic, "events_outline": events_outline})

    def plan_introduction(self, events_outline: str) -> str:
        chain = LLMChain(
            llm=self._llm,
            prompt=load_prompt(app_settings.STORY_DESIGNER_INTRODUCTION_PLANNER_PROMPT_PATH),
            verbose=app_settings.STORY_DESIGNER_VERBOSE,
        )
        return chain.run({"events_outline": events_outline})

    def write_introduction(self, introduction: str) -> str:
        chain = LLMChain(
            llm=self._llm,
            prompt=load_prompt(app_settings.STORY_DESIGNER_INTRODUCTION_WRITER_PROMPT_PATH),
            verbose=app_settings.STORY_DESIGNER_VERBOSE,
        )
        return chain.run(introduction)
