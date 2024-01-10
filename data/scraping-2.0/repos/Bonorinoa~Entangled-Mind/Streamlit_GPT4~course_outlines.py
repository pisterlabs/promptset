import openai
import streamlit as st


openai.api_key = st.secrets["OPENAI_KEY"]


def compute_cost(tokens, engine):
    
    model_prices = {"text-davinci-002": 0.02, 
                    "gpt-3.5-turbo": 0.002, 
                    "gpt-4": 0.03}
    model_price = model_prices[engine]
    
    cost = (tokens / 1000) * model_price

    return cost


def generate_outline(context, engine, max_tokens, temperature):
    
    prompt = f"Generate an outline for an online course on {context}"
    
    if st.session_state.engine == "text-davinci-002":
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        outline = response.choices[0].text.strip()
        tokens_used = response.usage.total_tokens
        query_cost = compute_cost(tokens_used, engine)
        
    else:
        response = openai.ChatCompletion.create(
        model=engine,
        messages=[{"role": "system", "content": "You are a helpful teaching assistant who is skilled in drafting outlines, both in English and Spanish, for succesful online courses."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        )
        outline = response['choices'][0]['message']['content'].strip()
        tokens_used = response.usage.total_tokens
        query_cost = compute_cost(tokens_used, engine)
        
    return (outline, tokens_used, query_cost)

# Define session state variables
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 256
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "engine" not in st.session_state:
    st.session_state.engine = "text-davinci-002"
if "tokens_used" not in st.session_state:
    st.session_state.tokens_used = 0
if "cost" not in st.session_state:
    st.session_state.cost = 0

# Set app title and instructions
st.title("Online Course Outline Generator")
st.write("Enter some context for your online course, and we'll generate an outline for you!")

# Add a sidebar for controlling model parameters
with st.sidebar:
    st.write("## Model Parameters")
    st.session_state.max_tokens = st.slider("Max Tokens", min_value=32, max_value=1000, value=st.session_state.max_tokens)
    st.session_state.temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=st.session_state.temperature, step=0.05)
    
    st.write("## Model Selection")
    st.session_state.engine = st.selectbox("Engine", 
                                           ["text-davinci-002", "gpt-3.5-turbo", "gpt-4"])

    st.write("### Model Prices")
    st.markdown("- (GPT-3) text-davinci-002: $0.02 per 1000 tokens")
    st.markdown("- (chatGPT-3.5) gpt-3.5-turbo: $0.002 per 1000 tokens")
    st.markdown("- (chatGPT-4) gpt-4: $0.03 per 1000 tokens")

    st.write("## Session Stats")

    st.write("## Tokens Used So Far...")
    st.write(st.session_state.tokens_used)
    
    st.write("## Cost So Far...")
    st.info(f"${st.session_state.cost:.5f}")
    
# Get user input and generate outline
context = st.text_input("Enter context for your course here:")
if context:
    try:
        if st.button("Generate Outline"):

            outline, tokens_used, query_cost = generate_outline(context, 
                                        st.session_state.engine, 
                                        st.session_state.max_tokens, 
                                        st.session_state.temperature)
            
            st.session_state.tokens_used += tokens_used
            st.session_state.cost += query_cost
            
            st.write("## Generated Outline")
            st.write(f"\n {outline} \n")
            st.info(f"Query Tokens: {tokens_used}. Total tokens used: {st.session_state.tokens_used}")
            st.info(f"Query Cost: {query_cost}. Total Session's Cost ${st.session_state.cost:.5f}")
            
            st.download_button("Download Outline", 
                               outline, 
                               file_name=f"{st.session_state.engine}_outline.txt")
            
    except Exception as e:
        st.error(e)
        
if not context:
    st.info("Please enter some context for your online course in the text box")