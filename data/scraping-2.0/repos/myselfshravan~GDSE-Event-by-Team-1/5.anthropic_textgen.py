from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import streamlit as st

# Set your Anthropic API key here
ANTHROPIC_API_KEY = st.secrets["calude_api_key"]
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

st.set_page_config(page_title="Anthropic Text Gen", page_icon="🤖")

st.title("AI Text Gen 🤖")
st.subheader("Write the Prompt")

prompt = st.text_area("Prompt", "Enter the Prompt Here")
pre_prompt = "Be creative! and explain in monkey language"

if st.button("Ask the AI"):
    completion = anthropic.completions.create(
        model="claude-1",
        max_tokens_to_sample=100,
        prompt=f"{HUMAN_PROMPT} {pre_prompt} {prompt} {AI_PROMPT}",
    )
    out = completion.completion
    st.write(out)
