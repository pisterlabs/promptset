import traceback
import openai
import os
from triple_quote_clean import TripleQuoteCleaner

try:
    import paragpt as sg
except ModuleNotFoundError:
    import sys
    import pathlib as pt

    sys.path.append((pt.Path(__file__).parent / "src").as_posix())
    import paragpt as sg

import etl
import streamlit as st


tqc = TripleQuoteCleaner()

extractor = sg.utils.Pipeline(lambda x: x.read().decode("utf8").replace("\r\n", "\n"))

developing = False

if developing:
    api_key = os.getenv("OPENAI_API_KEY")
    stage_1_cache = "stage1.parquet"
    stage_2_cache = "stage2.md"
else:
    api_key = ""
    stage_1_cache = None
    stage_2_cache = None

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns(3, gap="medium")


def main():
    with col1:
        st.title("Teams VTT Transcript Paraphraser")
        st.write("Paraphrase a Microsoft Teams VTT file Using Open AI")

        st.subheader("OpenAI API Key")
        open_api_key = st.text_input(
            "open api key", api_key, label_visibility="collapsed", type="password"
        )
        openai.api_key = open_api_key

        st.subheader("Paraphraser Model")
        paraphraser_model = st.text_input(
            "paraphraser model", "gpt-3.5-turbo", label_visibility="collapsed"
        )

        st.subheader("System Prompt")
        st.markdown(
            tqc
            << """
            Initialize every conversation with this prompt.
            This prompt provides context for the AI, such that it completes the task over the necessary domain
            """
        )
        system_prompt = st.text_area(
            "system prompt", etl._system_prompt, label_visibility="collapsed"
        )

        st.subheader("Paraphraser Prompt")
        st.write(
            tqc
            << """
            This prompt prepends every paraphrasable chunk
            """
        )
        summarization_prompt = st.markdown(
            "```\n" + sg.transformation.prompt_conversation_paraphrase.prompt + "\n```",
        )

    with col2:
        st.subheader("Start Chunking")
        st.write(
            """
                Specifies the number of tokens required to initiate  chunking.
            """
        )
        start_chunking = st.number_input(
            "start chunking",
            2800,
            label_visibility="collapsed",
        )

        st.subheader("Chunk Size")
        st.write(
            """
                Decides the max number of tokens within each chunked group
            """
        )
        max_tokens_per_chunk = st.number_input(
            "chunk size",
            1500,
            label_visibility="collapsed",
        )

        st.subheader("Train of Thought")
        st.write(
            "Train of thought used to increase the probability the AI performs the desired paraphrasing task"
        )
        txt = st.write(etl.train_of_thought)

    with col3:
        st.subheader("VTT File")
        uploaded_file = st.file_uploader(
            "Drag and drop a VTT file to clean and paraphrase ", type=["vtt"]
        )

        if uploaded_file is not None:
            content = extractor(uploaded_file)
            if len(content) > 100:
                st.markdown("Preview\n```\n" + content[:200] + "...\n```")
            else:
                st.markdown("```\n" + content[:200] + "\n```")

        if open_api_key == "":
            st.markdown("To run the paraphraser `provide an OpenAI API key`")
        elif uploaded_file is None:
            st.markdown("To run the paraphraser `provide VTT file`")
        else:
            st.markdown("Click `run` to start paraphrasing")
            if st.button("run", type="primary", use_container_width=True):
                # try:
                with st.spinner("Waiting for paraphraser to complete..."):
                    paraphrased = etl.load(
                        content,
                        paraphraser_model=paraphraser_model,
                        system_prompt=system_prompt,
                        stage_1_cache=stage_1_cache,
                        stage_2_cache=stage_2_cache,
                        max_tokens_per_chunk=max_tokens_per_chunk,
                        start_chunking=start_chunking,
                    )
                st.success("Paraphrase complete")
                st.subheader("Paraphrased Text")
                st.text_area(
                    "paraphrased",
                    paraphrased,
                    label_visibility="collapsed",
                    height=300,
                )
                # except Exception as e:
                #     st.error(
                #         f"Something Went Wrong!! {traceback.format_exc()}", icon="🚨"
                #     )

    return


if __name__ == "__main__":
    main()
