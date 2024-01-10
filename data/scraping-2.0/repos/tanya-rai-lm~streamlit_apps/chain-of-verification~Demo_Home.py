import asyncio
import os
import openai
import streamlit as st
import pandas as pd
from aiconfig import AIConfigRuntime

# streamlit page setup
st.set_page_config(
    page_title="Chain-of-Verification Template 🔗✅",
    page_icon="㏐",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chain-of-Verification Template 🔗 ✅")
st.subheader("Reduce Hallucinations from LLMs in 3 steps!")
st.markdown(
    """
    Large Language Models (LLMs) have been found to generate false information (hallucinations) even for facts that we and the model 
    know to be true. The Chain-of-Verification technique ensures that LLMs verify each claim they make, one-by-one, which results in improved accuracy. 
    This demo is made with [AIConfig](https://github.com/lastmile-ai/aiconfig).
    **[Link to Meta AI Research Paper](https://arxiv.org/pdf/2309.11495).**
    """
)
openai_api_key = st.text_input("Enter you OpenAI API Key to begin. Requires GPT-4:  🔑", type="password")

# chain-of-verification (cove) pipeline
async def cove():
    st.markdown("### ➡️ Step 1: Generate a numbered list of facts")
    st.markdown("""
        **Example Prompts:**
        1. Name 10 NBA players with more than 3 MVP (finals or regular season) awards. 
        2. Name 15 celebrities born in Toronto, Canada. 
        3. Name 20 programming languages developed in the USA.
    """)
    prompt = st.text_input(label="Write a prompt to generate a numbered list of 10-20 facts (like examples above).", placeholder="Ex:  Name 10 NBA players with more than 3 MVP (finals or regular season) awards.")
    if len(prompt) != 0:
        params = {"baseline_prompt": prompt}
        config.update_parameter("baseline_prompt", prompt)
        config.save()
        await config.run("baseline_response_gen", params)
        baseline_response_text = config.get_output_text("baseline_response_gen")
        config.get_prompt("baseline_response_gen").outputs = []
        st.write(baseline_response_text)

        st.markdown("### 💬 Step 2: Validate each fact individually")
        st.markdown("""
            **Example Questions:**
            1. How many MVP awards does this NBA player have?
            2. Where was this celebrity born?
            3. Where was this programming language developed? 
        """)
        verification_question = st.text_input(label="Enter a verification question to validate each fact. Follow the format below.", placeholder="Ex: How many MVP awards does this NBA player have?")

        entities = []
        verification_data = ""
        if len(verification_question)!=0:
            config.update_parameter("verification_question", verification_question + ": {{entity}}")
            config.save()
            rows = baseline_response_text.split('\n')
            for row in rows:
                if not row.strip():
                    continue
                entities.append(pd.Series(row).str.extract(r'(\d+\.\s)([^,]*)')[1].values[0])

            for n in entities:
                params = {"verification_question": verification_question, "entity": n}
                await config.run("verification", params)
                single_verification_text = config.get_output_text("verification")
                verification_data += " " + single_verification_text
                st.write(single_verification_text)
                print("\n")

            st.markdown("### ✅ Step 3: Revise the original response")
            params = {"verification_results": verification_data, "baseline_response_output": baseline_response_text}
            with st.spinner('Running Chain-of-Verification...'):
                await config.run("final_response_gen", params)
            st.markdown(config.get_output_text("final_response_gen"))


# aiconfig setup 
if openai_api_key:
    openai.api_key = openai_api_key
    config = AIConfigRuntime.load(os.path.join(os.path.dirname(__file__), "cove_aiconfig.json"))
    asyncio.run(cove())
