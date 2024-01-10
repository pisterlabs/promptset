import streamlit as st
import analysis_prompts as prompts
import os
from langchain.llms import Ollama
from langchain.llms import OpenAI

st.write("Example demo app on how to use use local Ollama models.")

st.write("Note on [\"Hallucinations\"](https://x.com/karpathy/status/1733299213503787018?s=46)")
st.write("If you refresh the page, new/different results will be generated")

if not os.getenv("USE_OPEN_AI"):
    model = st.selectbox('Select Model', ('mistral', 'llama2'), index=0)
    st.write("You have selected: " + model)

# Reset UI containers on each page reload - reload happens when a UI event is triggered (like a new model selection)
message_container = st.empty()
patch_container = st.empty()
analysis_container = st.empty()
yaml_container = st.empty()
summary_container = st.empty()
testcase_container = st.empty()
curl_container = st.empty()
code_container = st.empty()

# Loads model into RAM
# This is using Facebook's llama model (7B params - i.e. the smallest model available)
# It takes about 4GB of RAM to run.
# Larger models will perform better.
# When using this model, calls will be made to localhost:11434/ but will not be sent to any remote endpoints
# https://python.langchain.com/docs/integrations/llms/ollama
# Might also be worth looking at https://python.langchain.com/docs/integrations/chat/ollama
# Make sure to run: `export OPENAI_API_KEY=...` if using openAI
llm = OpenAI() if os.getenv("USE_OPEN_AI") else Ollama(model=model)

# Loads an example API yaml file
# You'd probably want to split this up into chunks before sending it to the model
# as the model will have a limit on how much context it can handle
dir_path = os.path.dirname(os.path.realpath(__file__))
yaml = open(os.path.join(dir_path, 'example_api.yaml'), 'r').read()
patch = open(os.path.join(dir_path, 'example.patch'), 'r').read()

with message_container.container():
    with st.spinner(text="Generating Message...",  cache=True):
        # Use the model to generate a very important message
        message = prompts.important_message(llm)

    with st.expander("Show Important message"):
        st.markdown(message)

with patch_container.container():
    with st.expander("Show patch"):
        st.markdown("```" + patch + "```")

with analysis_container.container():
    with st.spinner(text="Generating Patch Analysis...",  cache=True):
        analysis = prompts.code_patch_analysis(llm, patch)

    with st.expander("Show Analysis message"):
        st.markdown(analysis)

with yaml_container.container():
    with st.expander("Show API yaml"):
        st.markdown("```" + yaml + "```")

with summary_container.container():
    with st.spinner(text="Generating Summary...",  cache=True):
        # Use the model to generate a summary of the yaml.
        summary = prompts.summarize_api(llm, yaml)

    with st.expander("Show Summary"):
        st.markdown(summary)

with testcase_container.container():
    with st.spinner(text="Generating Test cases...",  cache=True):
        # Use the summary from previous stage to generate a set of test cases
        # You could pass the yaml in here to see if you get better results
        test_cases = prompts.describe_test_cases(llm, summary)

    with st.expander("Show Test Cases"):
        st.markdown(test_cases)

with curl_container.container():
    with st.spinner(text="Generating curl example...",  cache=True):
        # Use yaml to generate example curl & expected response
        example_curl = prompts.provide_example_curl_data(llm, yaml)

    with st.expander("Show curl data"):
        st.markdown(example_curl)

with code_container.container():
    with st.spinner(text="Generating code example...",  cache=True):
        # Use yaml to generate example kotlin batch upload request
        example_kotlin = prompts.batch_create_example(llm, yaml)

    with st.expander("Show kotlin code"):
        st.markdown(example_kotlin)




