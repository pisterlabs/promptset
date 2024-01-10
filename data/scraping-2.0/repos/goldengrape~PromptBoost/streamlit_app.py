import streamlit as st 
import pandas as pd
import os
import openai 
from openai.embeddings_utils import get_embedding, cosine_similarity


filename="prompts_df.pkl"
if os.path.exists(filename):
    prompts_df = pd.read_pickle(filename)

st.title("Prompt Boost")
st.write("This is a demo of Prompt Boost, a tool for generating better prompts.")
st.markdown("The basic idea of this demo is to have ChatGPT optimize a simple prompt into a better one. To do this, I will first find a well-crafted prompt example from a library that is similar in content to the user's prompt, ask ChatGPT to simplify it into a bad example, and then pair the bad-good examples to give to ChatGPT as examples for modifying the user's submitted prompt. ")
st.markdown("The well-crafted prompt example comes from [Awesome Chatgpt Prompts](https://github.com/f/awesome-chatgpt-prompts)")

open_ai_key=st.text_input("OpenAI API Key", value="", type="password")
query_prompt=st.text_area("Your Prompt", value="")
boost_button=st.button("Boost")

if open_ai_key != "" and query_prompt != "" and boost_button:
    with st.spinner("Generating Prompt..."):
        openai.api_key = open_ai_key
        embedding_model = "text-embedding-ada-002"
        query_embedding = get_embedding(query_prompt, engine=embedding_model)
        prompts_df["similarity"] = prompts_df["embedding"].apply(lambda x: cosine_similarity(x, query_embedding))

        results = prompts_df.sort_values("similarity", ascending=False, ignore_index=True)
        results = results.head(1)
        good_sample=results["good prompt"][0]
        good_sample=good_sample.split('","')[1]

        make_bad_sample_prompt=f"Simplify the excellent prompt below into a single sentence of no more than 10 words with ambiguous intent.\n\n{good_sample}"

        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
            messages=[{"role": "user", 
                        "content": make_bad_sample_prompt}]
            )
        
        bad_sample = (
                    completion["choices"][0]
                    .get("message")
                    .get("content")
                    .encode("utf8")
                    .decode()
                )
        bad_sample=bad_sample.lstrip("\n")
        
        prompt_boost_prompt=f"""
        I want you to act as prompt engineer. You are given a prompt and you need to make it better.
        You can use the following tips to make the prompt better:
        Clarity. Clear and concise prompts will help ensure that ChatGPT understands the topic or task at hand and is able to generate appropriate responses. Avoid using overly complex or ambiguous language, and be as specific as possible in your prompts.
        Focus. A clear prompt should have a clear purpose and focus, helping to guide the conversation and keep it on track. Avoid using overly broad or open-ended prompts, which may result in disjointed or unfocused conversations.
        Relevance. Ensure that your prompts are relevant to the user and conversation at hand. Avoid introducing unrelated topics or tangents that could distract from the main focus of the conversation.
        
        Here are the examples of the bad prompt and the good prompt below.
        
        bad prompt sample: 
        {bad_sample}
        
        good prompt sample:
        {good_sample}"
        
        The user prompt is: 
        '{query_prompt}'

        Please provide and only provide your optimized prompt.
        And write your prompt in English.
        """

        completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", 
                                "content": prompt_boost_prompt}]
                    )
        prompt_boost_result= (
                    completion["choices"][0]
                    .get("message")
                    .get("content")
                    .encode("utf8")
                    .decode()
                )
        prompt_boost_result=prompt_boost_result.lstrip("\n")
        st.text_area("result", value=prompt_boost_result, height=200)