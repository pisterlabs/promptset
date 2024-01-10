import logging
import time

from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _
from jsonfield import JSONField
from langchain.callbacks import get_openai_callback
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import TokenTextSplitter
from model_utils import Choices
from model_utils.models import TimeStampedModel

from .llm_tools import num_tokens_from_string
from .prompts import (
    letter_categorization,
    letter_evaluation_intro,
    letter_response_normalization,
    monitoring_response_normalized_template,
)

logger = logging.getLogger(__name__)


class LLmRequestQuerySet(models.QuerySet):
    def queued(self):
        return self.filter(status=self.model.STATUS.queued)


class LlmRequest(TimeStampedModel):
    STATUS = Choices(
        (0, "created", _("Created")),
        (1, "queued", _("Queued")),
        (2, "done", _("Done")),
        (3, "failed", _("Failed")),
    )
    engine_name = models.CharField(
        max_length=20, verbose_name=_("LLM Engine name"), null=True, blank=True
    )
    status = models.IntegerField(choices=STATUS, default=STATUS.created)
    request_prompt = models.TextField(
        verbose_name=_("LLM Engine request"), null=True, blank=True
    )
    response = models.TextField(
        verbose_name=_("LLM Engine response"), null=True, blank=True
    )
    token_usage = JSONField(
        verbose_name=_("LLM Engine token usage"), null=True, blank=True
    )
    objects = LLmRequestQuerySet.as_manager()

    class Meta:
        abstract = True

    def get_cost(self):
        if self.token_usage:
            return self.token_usage.get("total_cost", 0)
        return 0

    def get_time_used(self):
        if self.token_usage:
            return self.token_usage.get("completion_time", 0)
        return 0


class LlmLetterRequest(LlmRequest):
    evaluated_letter = models.ForeignKey(
        "letters.Letter",
        on_delete=models.DO_NOTHING,
        verbose_name=_("Evaluated Letter"),
    )

    @classmethod
    def categorize_letter(cls, letter):
        institution_name = ""
        monitoring_template = ""
        if letter.case and letter.case.monitoring:
            institution_name = letter.case.institution.name
            monitoring_template = letter.case.monitoring.template
        intro = letter_evaluation_intro.format(
            institution=institution_name,
            monitoring_question=monitoring_template,
        )
        test_prompt = letter_categorization.format(
            intro=intro,
            institution=institution_name,
            monitoring_response="",
        )
        # llm_engine = settings.OPENAI_API_ENGINE_35
        llm_engine = settings.OPENAI_API_ENGINE_4
        q_tokens = num_tokens_from_string(test_prompt, llm_engine)
        # print(f"q_tokens: {q_tokens}")

        max_tokens = min(settings.OPENAI_API_ENGINE_4_MAX_TOKENS, 6000) - q_tokens - 500
        # print(f"max_tokens: {max_tokens}")
        text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=100)
        texts = text_splitter.split_text(letter.get_full_content())
        # print(
        #     "texts[0] tokens:",
        #     num_tokens_from_string(texts[0], llm_engine),
        # )
        final_prompt = letter_categorization.format(
            intro=intro,
            institution=institution_name,
            monitoring_response=texts[0],
        )
        letter_llm_request = cls.objects.create(
            evaluated_letter=letter,
            engine_name=llm_engine,
            request_prompt=final_prompt,
            status=cls.STATUS.created,
            response="",
            token_usage={},
        )
        letter_llm_request.save()
        model = AzureChatOpenAI(
            openai_api_type=settings.OPENAI_API_TYPE,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_version=settings.OPENAI_API_VERSION,
            openai_api_base=settings.OPENAI_API_BASE,
            deployment_name=llm_engine,
            temperature=settings.OPENAI_API_TEMPERATURE,
        )
        chain = letter_categorization | model | StrOutputParser()
        start_time = time.time()
        with get_openai_callback() as cb:
            resp = chain.invoke(
                {
                    "intro": intro,
                    "institution": institution_name,
                    "monitoring_response": texts[0],
                }
            )
        end_time = time.time()
        execution_time = end_time - start_time
        llm_info_dict = vars(cb)
        llm_info_dict["completion_time"] = execution_time
        letter_llm_request.response = resp
        letter_llm_request.token_usage = llm_info_dict
        letter_llm_request.status = cls.STATUS.done
        letter_llm_request.save()
        letter.ai_evaluation = resp
        letter.save()
        # TODO: add case.response_received update
        # print(f"resp: {resp}")
        # print(f"cb: {cb}")
        # print(f"execution_time: {execution_time}")

    @classmethod
    def get_normalized_answers(cls, letter):
        institution_name = ""
        normalized_questions_json = ""
        if letter.case and letter.case.monitoring:
            institution_name = letter.case.institution.name
            if not letter.case.monitoring.normalized_response_template:
                logger.warning(
                    "Can not get normalised answer: normalized_response_template"
                    + f" missing in monitoring {letter.case.monitoring.pk}"
                )
                return
            if not letter.case.monitoring.use_llm:
                logger.warning(
                    "Skipping normalised answer: use_llm is False in monitoring"
                    + f" {letter.case.monitoring.pk}"
                )
                return
            normalized_questions_json = (
                letter.case.monitoring.normalized_response_template
            )
        else:
            logger.warning(
                f"Can not get normalised answer: letter {letter.pk}"
                + " has no case or monitoring."
            )
            return
        test_prompt = letter_response_normalization.format(
            institution=institution_name,
            normalized_questions=normalized_questions_json,
            monitoring_response="",
        )
        llm_engine = settings.OPENAI_API_ENGINE_35
        q_tokens = num_tokens_from_string(test_prompt, llm_engine)
        # print(f"q_tokens: {q_tokens}")

        max_tokens = (
            min(settings.OPENAI_API_ENGINE_35_MAX_TOKENS, 10000) - q_tokens - 500
        )
        # print(f"max_tokens: {max_tokens}")
        text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=100)
        texts = text_splitter.split_text(letter.get_full_content())
        # print(
        #     "texts[0] tokens:",
        #     num_tokens_from_string(texts[0], llm_engine),
        # )
        model = AzureChatOpenAI(
            openai_api_type=settings.OPENAI_API_TYPE,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_version=settings.OPENAI_API_VERSION,
            openai_api_base=settings.OPENAI_API_BASE,
            deployment_name=llm_engine,
            temperature=settings.OPENAI_API_TEMPERATURE,
        )
        chain = letter_response_normalization | model | StrOutputParser()
        for text in texts:
            final_prompt = letter_response_normalization.format(
                institution=institution_name,
                normalized_questions=normalized_questions_json,
                monitoring_response=text,
            )
            letter_llm_request = cls.objects.create(
                evaluated_letter=letter,
                engine_name=llm_engine,
                request_prompt=final_prompt,
                status=cls.STATUS.created,
                response="",
                token_usage={},
            )
            letter_llm_request.save()
            start_time = time.time()
            with get_openai_callback() as cb:
                resp = chain.invoke(
                    {
                        "institution": institution_name,
                        "normalized_questions": normalized_questions_json,
                        "monitoring_response": text,
                    }
                )
            end_time = time.time()
            execution_time = end_time - start_time
            llm_info_dict = vars(cb)
            llm_info_dict["completion_time"] = execution_time
            letter_llm_request.response = resp
            letter_llm_request.token_usage = llm_info_dict
            letter_llm_request.status = cls.STATUS.done
            letter_llm_request.save()
            normalized_questions_json = resp
        letter.normalized_response = normalized_questions_json
        letter.save()
        return normalized_questions_json


class LlmMonitoringRequest(LlmRequest):
    evaluated_monitoring = models.ForeignKey(
        "monitorings.Monitoring",
        on_delete=models.DO_NOTHING,
        verbose_name=_("Evaluated Monitoring"),
    )

    @classmethod
    def get_response_normalized_template(cls, monitoring):
        if not monitoring.use_llm:
            logger.info(
                f"Monitoring pk={monitoring.pk} not using LLM - skipping normalization."
            )
            return
        final_prompt = monitoring_response_normalized_template.format(
            monitoring_template=monitoring.template,
        )
        llm_engine = settings.OPENAI_API_ENGINE_35
        monitoring_llm_request = cls.objects.create(
            evaluated_monitoring=monitoring,
            engine_name=llm_engine,
            request_prompt=final_prompt,
            status=cls.STATUS.created,
            response="",
            token_usage={},
        )
        monitoring_llm_request.save()
        model = AzureChatOpenAI(
            openai_api_type=settings.OPENAI_API_TYPE,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_version=settings.OPENAI_API_VERSION,
            openai_api_base=settings.OPENAI_API_BASE,
            deployment_name=llm_engine,
            temperature=settings.OPENAI_API_TEMPERATURE,
        )
        chain = monitoring_response_normalized_template | model | StrOutputParser()
        start_time = time.time()
        with get_openai_callback() as cb:
            resp = chain.invoke({"monitoring_template": monitoring.template})
        end_time = time.time()
        execution_time = end_time - start_time
        llm_info_dict = vars(cb)
        llm_info_dict["completion_time"] = execution_time
        monitoring_llm_request.response = resp
        monitoring_llm_request.token_usage = llm_info_dict
        monitoring_llm_request.status = cls.STATUS.done
        monitoring_llm_request.save()
        monitoring.normalized_response_template = resp
        monitoring.save()
