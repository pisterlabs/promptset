"""Create a conversation chain for interview."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationChain
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory
from interview_sections import InterviewSections 
import logging
from typing import Any, Dict, List
from langchain.chains import SimpleSequentialChain


from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

def get_chain(interview_section, stream_handler, prompt, tracing: bool = False) -> ConversationChain:
    manager = AsyncCallbackManager([])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    streaming_llm_for_intro = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0.7,
        model = "gpt-4"
    )

    streaming_llm_for_interview= ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0.3,
        model = "gpt-4"
    )

    feedback_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0.7,
        model = "gpt-4"
    )

    # Share memory across agents 
    memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0), max_token_limit=1000)

    print("creating chain")

    if interview_section == InterviewSections.CODING_INTERVIEW_INTRO or interview_section == InterviewSections.CODING_INTERVIEW_OUTRO:
        conversation_chain = ConversationChain(
            prompt = prompt,
            llm=streaming_llm_for_intro, 
            verbose=True, 
            memory=memory
        )
    
    elif interview_section == InterviewSections.CODING_INTERVIEW_QUESTION_INTRO or interview_section == InterviewSections.CODING_INTERVIEW or interview_section== InterviewSections.CODING_INTERVIEW_CONCLUSION or interview_section == InterviewSections.CODING_INTERVIEW_FEEDBACK:
        conversation_chain = ConversationChain(
            prompt = prompt,
            llm=streaming_llm_for_interview, 
            verbose=True, 
            memory=memory
        )
    else:
        return None

    return conversation_chain

    #     elif interview_section == InterviewSections.CODING_INTERVIEW:
    #     constitutional_chain = ConstitutionalChain.from_llm(
    #         chain=streaming_llm_for_interview,
    #         constitutional_principles=[interview_ethics_principle],
    #         llm=streaming_llm_for_interview,
    #         verbose=True,
    #         memory=memory)
    
    # elif interview_section == InterviewSections.CODING_INTERVIEW_CONCLUSION:
    #     constitutional_chain = ConstitutionalChain.from_llm(
    #         chain=streaming_llm_for_interview,
    #         constitutional_principles=[interview_ethics_principle],
    #         llm=streaming_llm_for_interview,
    #         verbose=True,
    #         memory=memory)

    # elif interview_section == InterviewSections.CODING_INTERVIEW_OUTRO:
    #     constitutional_chain = ConstitutionalChain.from_llm(
    #         chain=streaming_llm_for_interview,
    #         constitutional_principles=[interview_ethics_principle],
    #         llm=streaming_llm_for_interview,
    #         verbose=True,
    #         memory=memory)
