import streamlit as st
from trulens_eval import Tru, Feedback, TruLlama, OpenAI as fOpenAI
from trulens_eval.feedback import Groundedness
import openai
from openai import OpenAI
import numpy as np

import nest_asyncio
nest_asyncio.apply()

openai.api_key = st.secrets["OPENAI_API_KEY"]
openai_client = OpenAI()

class OpenAI_custom(fOpenAI):
    """
    From tutorial: 
    https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/expositional/frameworks/langchain/langchain_agents.ipynb#scrollTo=hnXeWFcPUaqk
    """
    def no_answer_feedback(self, question: str, response: str) -> float:
        return float(openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "Does the RESPONSE provide an answer to the QUESTION? Rate on a scale of 1 to 10. Respond with the number only."},
            {"role": "user", "content": f"QUESTION: {question}; RESPONSE: {response}"}
        ]
    ).choices[0].message.content) / 10

custom_no_answer = OpenAI_custom()

def build_tru_recorder(agent):
    provider = fOpenAI()
    # HONEST
    # Answer Relevance
    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="[METRIC] Answer Relevance"
    ).on_input_output()

    # Context Relevance
    context_selection = TruLlama.select_source_nodes().node.text
    f_qs_relevance = (
        Feedback(provider.qs_relevance_with_cot_reasons,
                name="[METRIC] Context Relevance")
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    )

    # Groundedness of reponse based on context
    grounded = Groundedness(groundedness_provider=provider)
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons,
                name="[METRIC] Groundedness"
                )
        .on(context_selection)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # HARMLESS
    f_insensitivity = Feedback(
        provider.insensitivity_with_cot_reasons,
        name="[METRIC] Insensitivity",
        higher_is_better=False,
    ).on_output()

    f_input_maliciousness = Feedback(
        provider.maliciousness_with_cot_reasons,
        name="[METRIC] Input Maliciousness",
        higher_is_better=False,
    ).on_input()


    f_output_maliciousness = Feedback(
        provider.maliciousness_with_cot_reasons,
        name="[METRIC] Output Maliciousness",
        higher_is_better=False,
    ).on_output()

    # HELPFUL
    f_coherence = Feedback(
        provider.coherence_with_cot_reasons, name="[METRIC] Coherence"
    ).on_output()

    f_input_sentiment = Feedback(
        provider.sentiment_with_cot_reasons, name="[METRIC] Input Sentiment"
    ).on_input()

    f_output_sentiment = Feedback(
        provider.sentiment_with_cot_reasons, name="[METRIC] Ouput Sentiment"
    ).on_input()

    # AGENT: Missing tools
    f_no_answer = Feedback(
        custom_no_answer.no_answer_feedback, name="[METRIC] Answers Question"
    ).on_input_output()

    tru_recorder = TruLlama(
        agent,
        app_id="Students Agent",
        feedbacks=[
            f_qa_relevance,
            #f_qs_relevance,
            f_groundedness,
            f_insensitivity,
            f_input_maliciousness,
            #f_output_maliciousness,
            #f_coherence,
            #f_input_sentiment,
            #f_output_sentiment,
            #f_no_answer
        ]
    )
    return tru_recorder