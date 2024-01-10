import streamlit as st
import openai 

def get_completion(que, model): 
    resp_model = openai.Completion.create(
            model= model ,
            prompt=que,
            max_tokens=1024
            )
    print(resp_model)
    return resp_model["choices"][0]["text"].strip()

model_tip = [
#     """
# Turbo is the same model family that powers ChatGPT. It is optimized for conversational chat input and output but does equally well on completions when compared with the Davinci model family. Any use case that can be done well in ChatGPT should perform well with the Turbo model family in the API.
# The Turbo model family is also the first to receive regular model updates like ChatGPT.
# \n
# **Good at: Conversation and text generation**
#     \n
#     """,
#     """
# Turbo is the same model family that powers ChatGPT. It is optimized for conversational chat input and output but does equally well on completions when compared with the Davinci model family. Any use case that can be done well in ChatGPT should perform well with the Turbo model family in the API.
# The Turbo model family is also the first to receive regular model updates like ChatGPT.
# \n
# **Good at: Conversation and text generation**
#     \n
#     """,
    """
Davinci is the most capable model family and can perform any task the other models (ada, curie, and babbage) can perform and often with less instruction. For applications requiring a lot of understanding of the content, like summarization for a specific audience and creative content generation, Davinci will produce the best results. These increased capabilities require more compute resources, so Davinci costs more per API call and is not as fast as the other models.
Another area where Davinci shines is in understanding the intent of text. Davinci is quite good at solving many kinds of logic problems and explaining the motives of characters. Davinci has been able to solve some of the most challenging AI problems involving cause and effect.
\n
**Good at: Complex intent, cause and effect, summarization for audience**
    \n
    """,
    """
Davinci is the most capable model family and can perform any task the other models (ada, curie, and babbage) can perform and often with less instruction. For applications requiring a lot of understanding of the content, like summarization for a specific audience and creative content generation, Davinci will produce the best results. These increased capabilities require more compute resources, so Davinci costs more per API call and is not as fast as the other models.
Another area where Davinci shines is in understanding the intent of text. Davinci is quite good at solving many kinds of logic problems and explaining the motives of characters. Davinci has been able to solve some of the most challenging AI problems involving cause and effect.
\n
**Good at: Complex intent, cause and effect, summarization for audience**
    \n
    """,
    """
Curie is extremely powerful, yet very fast. While Davinci is stronger when it comes to analyzing complicated text, Curie is quite capable for many nuanced tasks like sentiment classification and summarization. Curie is also quite good at answering questions and performing Q&A and as a general service chatbot.
\n
**Good at: Language translation, complex classification, text sentiment, summarization**
    \n
    """,
    """
Babbage can perform straightforward tasks like simple classification. It’s also quite capable when it comes to Semantic Search ranking how well documents match up with search queries.
\n
**Good at: Moderate classification, semantic search classification**
    \n
    """,
    """
Ada is usually the fastest model and can perform tasks like parsing text, address correction and certain kinds of classification tasks that don’t require too much nuance. Ada’s performance can often be improved by providing more context.
\n
**Good at: Parsing text, simple classification, address correction, keywords**
\n
Note: Any task performed by a faster model like Ada can be performed by a more powerful model like Curie or Davinci.
    """,
]
model_list = [
    # "gpt-3.5-turbo",
    # "gpt-3.5-turbo-0301",
    "text-davinci-003",
    "davinci",
    "curie",
    "babbage",
    "ada",
]


def question_answering():
    st.title("Question answering with openai models")

    model_selection = st.selectbox(
        "Select model from following openai models", options=model_list
    )
    if model_selection:
        # form for asking questions 
        q_sel_form = st.form("que_selection" , clear_on_submit=  False )
        question = q_sel_form.text_area("Enter your question here")
        # question submit button 
        submitted = q_sel_form.form_submit_button("Submit")     
        if submitted:
            answer = get_completion(question , model_selection) 
            st.markdown(f"""
            ### Answer: 
            """)
            st.code(answer) 
        # show model tip
        st.success(model_tip[model_list.index(model_selection)])
