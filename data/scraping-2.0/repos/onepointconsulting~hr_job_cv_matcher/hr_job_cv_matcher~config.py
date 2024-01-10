from pathlib import Path
import os

from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

from hr_job_cv_matcher.log_init import logger

load_dotenv()


def create_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


class Config:
    model = os.getenv("OPENAI_MODEL")
    request_timeout = int(os.getenv("REQUEST_TIMEOUT"))
    has_langchain_cache = os.getenv("LANGCHAIN_CACHE") == "true"
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        request_timeout=request_timeout,
        cache=has_langchain_cache,
    )
    remote_pdf_server = os.getenv("REMOTE_PDF_SERVER")
    temp_doc_location = Path(os.getenv("TEMP_DOC_LOCATION"))
    create_if_not_exists(temp_doc_location)
    max_jd_files = os.getenv("MAX_JD_FILES") or 10
    max_cv_files = os.getenv("MAX_CV_FILES") or 10
    verbose_llm = os.getenv("VERBOSE_LLM") == "true"


cfg = Config()


class PromptConfig:
    extra_skills = ""

    def update_prompt(self, settings):
        prompt_extra_skills_examples = settings["prompt_extra_skills_examples"]
        if (
            prompt_extra_skills_examples is not None
            and len(prompt_extra_skills_examples) > 0
        ):
            splits = prompt_extra_skills_examples.split(",")
            output = ""
            for split in splits:
                output += f"\n-{split.strip()}"
            self.extra_skills = output


prompt_cfg = PromptConfig()

if __name__ == "__main__":
    logger.info("Model: %s", cfg.model)
    logger.info("Remote_pdf_server: %s", cfg.remote_pdf_server)
    logger.info("Max CV Files: %s", cfg.max_cv_files)
