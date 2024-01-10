import streamlit as st
import openai


# Configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Store your API key as an environment variable for security
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables.")

openai.api_type = "azure"
openai.api_base = "https://example.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = OPENAI_API_KEY

def get_codex_response(prompt_text):
    """Get response from Codex for the given prompt."""
    try:
        response = openai.Completion.create(
            engine="code",
            prompt=prompt_text,
            temperature=0.7,  # Adjusted for slightly more deterministic output
            max_tokens=2000,  # Increased for longer responses
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            stop=["Human:", "AI:"]
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def main():
    st.title("Hello karl using OpenAI Codex")
    
    # User input
    name = st.text_input("Describe the functionality you want in the code (e.g. 'a function to sort a list of numbers')")
    
    if name:
        # Provide feedback while API call is made
        with st.spinner("Generating code..."):
            prompt_text = f"\"\"\"\nWrite a detailed Python function for: {name}\n\"\"\""
            code_response = get_codex_response(prompt_text)
        
        if code_response:
            st.code(code_response, language='python')

if __name__ == "__main__":
    main()
