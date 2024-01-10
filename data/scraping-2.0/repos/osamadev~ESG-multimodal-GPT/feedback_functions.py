def load_feedback_functions():
    from trulens_eval.feedback.provider import OpenAI
    from trulens_eval import Feedback
    from trulens_eval import Select
    import numpy as np
    # Initialize provider class
    openai = OpenAI()
    from trulens_eval.feedback import Groundedness
    grounded = Groundedness(groundedness_provider=OpenAI())
    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(name="Groundedness", imp=grounded.groundedness_measure_with_cot_reasons)
        .on(Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(name="Answer Relevancy", imp=openai.relevance_with_cot_reasons).on_input_output()
    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(name="Context Relevancy", imp=openai.qs_relevance_with_cot_reasons)
        .on_input()
        .on(Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content)
        .aggregate(np.mean)
    )

    # Moderation metrics on output
    f_hate = Feedback(openai.moderation_hate).on_output()
    f_violent = Feedback(openai.moderation_violence, higher_is_better=False).on_output()
    f_selfharm = Feedback(openai.moderation_selfharm, higher_is_better=False).on_output()
    f_maliciousness = Feedback(openai.maliciousness_with_cot_reasons, higher_is_better=False).on_output()

    return (f_groundedness, f_qa_relevance, f_context_relevance, f_hate, f_violent, f_selfharm, f_maliciousness)